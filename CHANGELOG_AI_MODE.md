# Changelog - Tính năng chơi với AI

## Ngày cập nhật: 2025-11-22

### Các tính năng mới đã thêm:

#### 1. Màn hình chọn chế độ chơi
- Thêm `GameModeDialog` trong `ui_main.py`
- Cho phép người chơi chọn giữa 2 chế độ:
  - 🤖 **Chơi với AI** (1 người chơi)
  - 👥 **Hai người chơi** (2 người chơi)

#### 2. Màn hình nhập tên linh hoạt
- Cập nhật `PlayerNameDialog` để hỗ trợ cả 2 chế độ
- **Chế độ 1 người chơi:**
  - Chỉ hiển thị 1 ô nhập tên cho người chơi
  - Player 2 tự động được đặt là "AI"
- **Chế độ 2 người chơi:**
  - Hiển thị 2 ô nhập tên như cũ

#### 3. Phím tắt N thông minh
- **Chế độ 1 người chơi:** Phím N chỉ đổi tên cho 1 người chơi
- **Chế độ 2 người chơi:** Phím N đổi tên cho cả 2 người chơi

#### 4. Logic chơi với AI
**Luồng game khi chơi với AI:**

1. **Màn hình chờ (play mode):**
   - Bên trái: Hiển thị camera người chơi với gesture detection
   - Bên phải: Hiển thị ảnh `asset/bot-play/rule.jpg`

2. **Countdown (3, 2, 1):**
   - Khi đếm đến **1**, AI random ra kéo/búa/bao
   - Ảnh bên phải đổi thành kết quả của AI:
     - `asset/bot-play/rock.jpg` (Búa)
     - `asset/bot-play/paper.jpg` (Giấy)
     - `asset/bot-play/sisscors.jpg` (Kéo)

3. **Kết quả:**
   - So sánh gesture người chơi vs AI
   - Hiển thị người thắng
   - Cập nhật điểm số

**Mục đích:** Thử thách phản xạ người chơi - họ có 1 giây cuối để thay đổi quyết định sau khi AI ra tay!

#### 5. Cải tiến hệ thống

**Trong `ui_main.py`:**
- Thêm `game_mode` attribute vào `RPSApplication` để track "single" hoặc "two"
- Thêm `GameModeDialog` class mới
- Cập nhật `PlayerNameDialog` nhận parameter `mode`
- Cập nhật `GameWindow` nhận parameter `game_mode`

**Trong `main_gui.py`:**
- Load bot images từ `asset/bot-play/`
- Thêm `bot_gesture` để lưu quyết định của AI
- Đổi `game_mode` thành `game_mode_state` (để tránh xung đột)
- Cập nhật logic:
  - `update_frame()` xử lý cả 2 chế độ
  - Chỉ tạo `Player 2` khi ở chế độ 2 người chơi
  - Bot random gesture khi countdown = 1
  - Hiển thị bot image tương ứng với game state

### Các file đã chỉnh sửa:
1. `ui_main.py` - Thêm GameModeDialog, cập nhật PlayerNameDialog
2. `main_gui.py` - Thêm logic chơi với AI

### Các file cần thiết:
- `asset/bot-play/rule.jpg` - Ảnh hiển thị khi chưa chơi
- `asset/bot-play/rock.jpg` - Ảnh búa của AI
- `asset/bot-play/paper.jpg` - Ảnh giấy của AI  
- `asset/bot-play/sisscors.jpg` - Ảnh kéo của AI

### Hướng dẫn sử dụng:

1. Chạy game: `python main_gui.py`
2. Chọn chế độ chơi (1 người hoặc 2 người)
3. Nhập tên
4. Chơi game với các phím tắt:
   - **SPACE**: Bắt đầu vòng chơi
   - **R**: Reset điểm
   - **N**: Đổi tên (1 người nếu chơi với AI, 2 người nếu PvP)
   - **Q**: Thoát game
   - **F11**: Toggle fullscreen

### Cập nhật mới (Build 2):

#### 1. Tăng độ khó
- **Giảm thời gian chụp:** Từ `0.3s` xuống `0.15s`
- Người chơi có ít thời gian phản ứng hơn sau khi AI ra tay
- Game trở nên thử thách hơn và đòi hỏi phản xạ nhanh hơn

#### 2. Cải thiện UX/UI Navigation
- **Bỏ phím N (Rename):** Không còn chức năng đổi tên trong game
- **Thêm phím ESC (Menu):** Quay về màn hình chọn chế độ
- **Flow mới:**
  - Nhấn ESC → Về màn hình chọn chế độ
  - Chọn lại 1 người/2 người
  - Nhập tên mới
  - Bắt đầu game mới

#### 3. Cập nhật phím tắt
**Trước:**
- SPACE: Bắt đầu | R: Reset điểm | N: Đổi tên | Q: Thoát

**Sau:**
- SPACE: Bắt đầu | R: Reset điểm | ESC: Menu | Q: Thoát

### Known Issues:
- Không có

### Future Improvements:
- Có thể thêm độ khó cho AI (easy/medium/hard)
- Thêm animation cho bot gesture reveal
- Thêm sound effects riêng cho AI
- Thêm confirmation dialog khi nhấn ESC để tránh thoát nhầm
