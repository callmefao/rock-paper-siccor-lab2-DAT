# 🚀 Code Optimization Roadmap

## 📋 Tổng Quan
Dự án hiện tại có **7-9 threads** đang chạy đồng thời với một số vấn đề về thread safety, memory leaks, và performance. Document này liệt kê tất cả các optimization tasks cần thực hiện.

---

## 🔴 PRIORITY 1 - Critical Issues (Phải fix ngay)

### ✅ Task 1.1: Fix Race Condition trong Capture Logic **[COMPLETED]**
**File:** `main.py`, `main_gui.py`  
**Vấn đề:** Sử dụng `hasattr()` để check dynamic attributes - không thread-safe  
**Status:** ✅ Fixed in commit `4796f97` on branch `fix/race-condition`
```python
# Current (BAD)
if not hasattr(self, '_capture_done'):
    self._captured_frame_left = clean_frame_left.copy()
```

**Giải pháp:**
- Tạo instance variables cố định trong `__init__`
- Sử dụng dictionary hoặc dataclass để quản lý state
```python
def __init__(self):
    self.capture_state = {
        'done': False,
        'predictions_ready': False,
        'capture_time': None,
        'captured_frames': {'left': None, 'right': None}
    }
```

**Ước lượng:** 30 phút  
**Impact:** HIGH - Ngăn chặn crash khi spam SPACE

---

### ✅ Task 1.2: Fix Memory Leak trong AudioManager **[COMPLETED]**
**File:** `ui_main.py`  
**Vấn đề:** `fade_timer` không được cancel khi window đóng  
**Status:** ✅ Fixed in commit `8576427` on branch `fix/memory-leak-in-audio-manager`

**Giải pháp:**
```python
class AudioManager:
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.stop_all()
    
    def stop_all(self):
        if self.fade_timer:
            self.fade_timer.cancel()
            self.fade_timer = None
        pygame.mixer.music.stop()
        self.bg_music_playing = False
```

**Ước lượng:** 15 phút  
**Impact:** HIGH - Ngăn memory leak khi đóng/mở app nhiều lần

---

### ✅ Task 1.3: Fix Deadlock Risk trong Player._process_loop **[COMPLETED]**
**File:** `main.py` (dòng 85-130)  
**Vấn đề:** Lock được giữ quá lâu trong vòng lặp xử lý  
**Status:** ✅ Fixed in commit `5217d8b` on branch `fix/deadlock-risk`

**Giải pháp:**
```python
# BAD (current)
with self.lock:
    if self.frame is None:
        continue
    frame_to_process = self.frame
    current_game_mode = self.game_mode

# GOOD
with self.lock:
    frame_snapshot = self.frame
    mode_snapshot = self.game_mode

if frame_snapshot is None:
    continue
# Process outside lock
```

**Ước lượng:** 20 phút  
**Impact:** HIGH - Giảm lock contention, tăng FPS

---

## 🟠 PRIORITY 2 - Performance Issues (Tối ưu hiệu năng)

### ✅ Task 2.1: Loại Bỏ Redundant Frame Copying
**File:** `main.py`, `main_gui.py`  
**Vấn đề:** Tạo 4 bản copy mỗi frame không cần thiết

**Giải pháp:**
- Chỉ copy frames khi vào countdown mode (cần capture)
- Sử dụng frame slicing thay vì `.copy()` khi có thể
```python
# Chỉ copy khi cần
if game_mode == "countdown" and countdown_start is not None:
    clean_frame_left = frame_left.copy()
    clean_frame_right = frame_right.copy()
```

**Ước lượng:** 25 phút  
**Impact:** MEDIUM - Giảm 15-20% CPU usage  
**Benchmark:** 4 frame copies × 1280×360×3 bytes × 30 FPS = ~125 MB/s tiết kiệm

---

### ✅ Task 2.2: Optimize Frame Skip Counter
**File:** `main.py` (dòng 108-118)  
**Vấn đề:** Counter không reset, có thể overflow

**Giải pháp:**
```python
# Reset counter định kỳ
if frame_skip_counter > 1000000:
    frame_skip_counter = 0

# Hoặc dùng modulo trực tiếp
if (self.frame_count % 3) == 0:  # Countdown mode
    # Process frame
```

**Ước lượng:** 10 phút  
**Impact:** LOW - Chỉ ảnh hưởng sau nhiều giờ chạy

---

### ✅ Task 2.3: Implement ThreadPoolExecutor cho Sound Effects
**File:** `ui_main.py`  
**Vấn đề:** Mỗi sound effect tạo 1 thread mới → thread explosion khi spam

**Giải pháp:**
```python
from concurrent.futures import ThreadPoolExecutor

class AudioManager:
    def __init__(self):
        self.sound_executor = ThreadPoolExecutor(
            max_workers=3, 
            thread_name_prefix="AudioWorker"
        )
    
    def play_countdown_sound(self):
        self.sound_executor.submit(self._play_countdown_impl)
    
    def play_winner_sound(self, sound_file):
        self.sound_executor.submit(self._play_winner_impl, sound_file)
```

**Ước lượng:** 45 phút  
**Impact:** HIGH - Giới hạn max threads, tránh thread pool exhaustion

---

### ✅ Task 2.4: Reduce MediaPipe Instance Memory
**File:** `main.py` (Player class)  
**Vấn đề:** 2 MediaPipe instances = ~400MB RAM

**Giải pháp (Advanced):**
- Share 1 MediaPipe instance giữa 2 players
- Sử dụng queue để serialize requests
```python
class SharedMediaPipeProcessor:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(...)
        self.request_queue = Queue()
        self.result_queues = {1: Queue(), 2: Queue()}
    
    def process_frame(self, player_id, frame):
        self.request_queue.put((player_id, frame))
        return self.result_queues[player_id].get(timeout=0.1)
```

**Ước lượng:** 2 giờ  
**Impact:** MEDIUM - Tiết kiệm 200MB RAM nhưng có thể giảm FPS

---

## 🟡 PRIORITY 3 - Code Quality (Cải thiện chất lượng code)

### ✅ Task 3.1: Tạo Constants File
**File mới:** `constants.py`  
**Vấn đề:** Magic numbers và hardcoded paths ở khắp nơi

**Giải pháp:**
```python
# constants.py
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
ASSET_DIR = PROJECT_ROOT / "asset"
SOUND_DIR = ASSET_DIR / "sound"
ICON_DIR = ASSET_DIR / "icons"
MODEL_DIR = PROJECT_ROOT / "model"

# Audio Settings
NORMAL_BG_VOLUME = 0.15
FADE_BG_VOLUME = 0.05
WINNER_SOUND_VOLUME = 0.7
FADE_DURATION = 10.0

# UI Settings
THUMBNAIL_SIZE = 180
LOGO_HEIGHT = 80
CAPTURE_BOX_SIZE = 280

# Game Settings
DEFAULT_FPS = 30
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
COUNTDOWN_DURATION = 3
RESULT_DISPLAY_TIME = 3
```

**Ước lượng:** 30 phút  
**Impact:** MEDIUM - Dễ maintain, tránh typo

---

### ✅ Task 3.2: Remove Unused Imports
**File:** `ui_main.py`, `main_gui.py`, `main.py`  
**Vấn đề:** Import thư viện không dùng

**Danh sách cần loại bỏ:**
```python
# ui_main.py
from PyQt5.QtWidgets import (..., QSizePolicy)  # ← Không dùng
from PyQt5.QtGui import (..., QPalette, QColor)  # ← Không dùng

# main_gui.py
import sys  # Có thể trùng với import khác
```

**Ước lượng:** 15 phút  
**Impact:** LOW - Chỉ cleanup

---

### ✅ Task 3.3: Refactor God Class: RPSApplication
**File:** `ui_main.py`  
**Vấn đề:** Class làm quá nhiều việc (dialogs, audio, game logic)

**Giải pháp - Tách thành:**
```python
# dialog_manager.py
class DialogManager:
    def show_name_input(self) -> tuple[str, str]:
        pass
    def show_loading_screen(self, p1_name, p2_name):
        pass

# audio_service.py  
class AudioService:
    def __init__(self):
        self.bg_music = BackgroundMusicPlayer()
        self.sfx_player = SoundEffectPlayer()
        self.tts_generator = TTSGenerator()

# game_controller.py
class GameController:
    def __init__(self, dialog_mgr, audio_svc):
        self.dialog_manager = dialog_mgr
        self.audio_service = audio_svc
```

**Ước lượng:** 3 giờ  
**Impact:** HIGH - Dễ test, maintain, extend

---

### ✅ Task 3.4: Simplify Audio Stack
**File:** `ui_main.py`, `main.py`  
**Vấn đề:** Dùng 3 thư viện audio khác nhau (pygame, winsound, gTTS)

**Giải pháp:**
- Chỉ dùng **pygame** cho tất cả (hỗ trợ đầy đủ WAV, MP3, OGG)
```python
# Thay thế winsound
# OLD
winsound.PlaySound("countdown.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)

# NEW
countdown_sound = pygame.mixer.Sound("countdown.wav")
countdown_sound.play()
```

**Ước lượng:** 1 giờ  
**Impact:** MEDIUM - Giảm dependencies, code đơn giản hơn

---

### ✅ Task 3.5: Add Type Hints
**File:** Tất cả `.py` files  
**Vấn đề:** Không có type hints → khó debug

**Giải pháp:**
```python
# Before
def update_frame(self, frame, game_mode="play"):
    pass

# After
def update_frame(self, frame: np.ndarray, game_mode: str = "play") -> None:
    pass

# Class attributes
class Player:
    player_id: int
    name: str
    model: RidgeClassifier
    scaler: StandardScaler
    prediction_buffer: deque[Optional[str]]
```

**Ước lượng:** 2 giờ  
**Impact:** MEDIUM - Dễ maintain, IDE autocomplete tốt hơn

---

## 🟢 PRIORITY 4 - Architecture Improvements (Nâng cao kiến trúc)

### ✅ Task 4.1: Implement Observer Pattern cho Game Events
**File mới:** `events.py`  
**Vấn đề:** GameWindow phải biết quá nhiều về game logic

**Giải pháp:**
```python
# events.py
from dataclasses import dataclass
from typing import Protocol

@dataclass
class GameEvent:
    pass

@dataclass
class ScoreChangedEvent(GameEvent):
    player1_score: int
    player2_score: int
    draws: int

class GameEventListener(Protocol):
    def on_event(self, event: GameEvent) -> None: ...

class GameEventBus:
    def __init__(self):
        self.listeners: list[GameEventListener] = []
    
    def subscribe(self, listener: GameEventListener):
        self.listeners.append(listener)
    
    def publish(self, event: GameEvent):
        for listener in self.listeners:
            listener.on_event(event)
```

**Ước lượng:** 4 giờ  
**Impact:** HIGH - Loose coupling, dễ test

---

### ✅ Task 4.2: Separate Training Code from Runtime
**File:** Tạo package `training/`  
**Vấn đề:** `train.py` mix với game logic

**Giải pháp:**
```
project/
├── src/
│   ├── game/          # Runtime code
│   │   ├── main.py
│   │   ├── main_gui.py
│   │   ├── ui_main.py
│   └── shared/        # Shared utilities
│       └── hand_feature_extractor.py
└── training/          # Training code (separate)
    ├── train.py
    ├── dataset_loader.py
    └── augmentor.py
```

**Ước lượng:** 1.5 giờ  
**Impact:** MEDIUM - Clear separation of concerns

---

### ✅ Task 4.3: Add Configuration System
**File mới:** `config.yaml`, `config_loader.py`  
**Vấn đề:** Settings scatter trong code

**Giải pháp:**
```yaml
# config.yaml
camera:
  width: 1280
  height: 720
  fps: 30
  backend: "DSHOW"  # DirectShow for Windows

audio:
  background_volume: 0.15
  fade_volume: 0.05
  winner_volume: 0.7
  fade_duration: 10.0

game:
  countdown_duration: 3
  result_display_time: 3
  smoothing_buffer_size: 5

mediapipe:
  static_image_mode: false
  max_num_hands: 1
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5
```

**Ước lượng:** 2 giờ  
**Impact:** HIGH - Easy configuration without code changes

---

## 📊 PRIORITY 5 - Testing & Monitoring (Tùy chọn)

### ✅ Task 5.1: Add Thread Monitoring Dashboard
**File mới:** `monitor.py`  
**Mục đích:** Real-time thread count và CPU usage

**Giải pháp:**
```python
import threading
import psutil
import time

class ThreadMonitor:
    def monitor(self):
        while True:
            thread_count = threading.active_count()
            cpu_percent = psutil.cpu_percent(interval=1)
            mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
            
            print(f"Threads: {thread_count} | CPU: {cpu_percent}% | RAM: {mem_mb:.1f}MB")
            time.sleep(2)

# Usage in main
monitor = ThreadMonitor()
threading.Thread(target=monitor.monitor, daemon=True).start()
```

**Ước lượng:** 1 giờ  
**Impact:** LOW - Chỉ để debug

---

### ✅ Task 5.2: Add Unit Tests
**File mới:** `tests/` folder  
**Scope:** Test critical functions

**Danh sách test cases:**
```python
# test_feature_extraction.py
def test_normalize_hand_orientation():
    landmarks = np.random.rand(21, 3)
    result = normalize_hand_orientation(landmarks)
    assert result.shape == (21, 3)

# test_game_logic.py
def test_determine_winner():
    assert determine_winner("Búa", "Kéo") == "p1"
    assert determine_winner("Kéo", "Búa") == "p2"
    assert determine_winner("Búa", "Búa") == "draw"

# test_audio_manager.py
def test_volume_fade():
    audio_mgr = AudioManager()
    audio_mgr.fade_bg_music_down()
    # Assert volume changed
```

**Ước lượng:** 4 giờ  
**Impact:** MEDIUM - Prevent regressions

---

## 📈 Tổng Kết Thời Gian Ước Lượng

| Priority | Tasks | Tổng Thời Gian | Impact |
|----------|-------|----------------|--------|
| Priority 1 | 3 tasks | ~1.5 giờ | CRITICAL |
| Priority 2 | 4 tasks | ~4 giờ | HIGH |
| Priority 3 | 5 tasks | ~7.5 giờ | MEDIUM |
| Priority 4 | 3 tasks | ~7.5 giờ | HIGH |
| Priority 5 | 2 tasks | ~5 giờ | LOW |
| **TỔNG** | **17 tasks** | **~25 giờ** | |

---

## 🎯 Recommended Execution Order

### Phase 1: Critical Fixes (Làm trước, ~2 giờ)
1. Task 1.1 - Race condition fix
2. Task 1.2 - Memory leak fix
3. Task 1.3 - Deadlock fix
4. Task 2.1 - Remove redundant copies

### Phase 2: Thread Safety (Làm tiếp, ~1 giờ)
5. Task 2.3 - ThreadPoolExecutor
6. Task 2.2 - Frame skip counter

### Phase 3: Code Quality (Có thể dần dần, ~3-4 giờ)
7. Task 3.1 - Constants file
8. Task 3.2 - Remove unused imports
9. Task 3.4 - Simplify audio stack

### Phase 4: Architecture (Optional, khi có thời gian)
10. Task 3.3 - Refactor god class
11. Task 4.1 - Observer pattern
12. Task 4.3 - Configuration system

---

## ✅ Success Metrics

Sau khi hoàn thành optimization, dự án sẽ đạt:

- ✅ **Stability:** Không crash khi spam SPACE 100 lần/phút
- ✅ **Performance:** Giảm CPU usage 20-30%
- ✅ **Memory:** Giảm RAM usage 15-20%
- ✅ **Thread Count:** Giảm từ 7-9 → 5-6 threads
- ✅ **Maintainability:** Code coverage với type hints 80%+
- ✅ **FPS Stability:** Giữ 30 FPS ổn định, không jitter

---

## 📝 Notes

- Tất cả changes nên có git commit riêng với message rõ ràng
- Test kỹ sau mỗi task trước khi chuyển sang task tiếp theo
- Backup code trước khi làm refactoring lớn (Priority 3-4)
- Có thể skip Priority 5 nếu không cần thiết

---

**Last Updated:** 2025-11-19  
**Version:** 1.0  
**Status:** 📋 Planning Phase
