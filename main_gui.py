# =====================================
# Rock Paper Scissors - PyQt5 GUI Version
# =====================================
import cv2
import numpy as np
import sys
import time
import random
import json
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication
from joblib import load
from PIL import Image, ImageDraw, ImageFont
import os

# Import UI components
from ui_main import RPSApplication, GameWindow

# Import game components
from hand_feature_extractor import HandFeatureExtractor
from main import Player, determine_winner, play_sound


# =====================================
# Config Loader
# =====================================
def load_config(config_path="config.json"):
    """Load configuration from JSON file"""
    default_config = {
        "camera": {"width": 640, "height": 480, "fps": 30, "buffer_size": 1},
        "game": {"countdown_duration": 3, "result_display_time": 3, "target_fps": 30},
        "mediapipe": {"min_detection_confidence": 0.3, "min_tracking_confidence": 0.3, "processing_scale": 0.75},
        "ai_strategy": {
            "default_mode": "random",
            "cheat_threshold": 3,
            "auto_balance": {"enabled": True, "target_win_rate": 0.5, "check_interval": 5, "min_games_before_balance": 3},
            "admin_hotkey": "C"
        },
        "model": {"model_path": "model/rps_ridge_model.joblib", "scaler_path": "model/rps_scaler.joblib"},
        "ui": {"show_admin_indicator": True, "admin_indicator_size": "small", "logo_max_height": 40}
    }

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            print(f"✓ Config loaded from {config_path}")
            return config
    except FileNotFoundError:
        print(f"⚠ Config file not found, using defaults")
        return default_config
    except json.JSONDecodeError as e:
        print(f"⚠ Config JSON error: {e}, using defaults")
        return default_config


# =====================================
# Enhanced RPSGame with PyQt5 Integration
# =====================================
class RPSGameGUI:
    """Main game class with PyQt5 GUI integration"""

    def __init__(self, app_manager, config):
        """
        Initialize game with config

        Args:
            app_manager: RPSApplication instance
            config: Configuration dictionary loaded from config.json
        """
        self.app_manager = app_manager
        self.audio_manager = app_manager.audio_manager
        self.game_mode_type = app_manager.game_mode  # "ai" or "pvp"
        self.config = config

        # Load trained model and scaler from config
        model_path = config["model"]["model_path"]
        scaler_path = config["model"]["scaler_path"]
        self.model = load(model_path)
        self.scaler = load(scaler_path)

        # Game configuration from config
        self.camera_width = config["camera"]["width"]
        self.camera_height = config["camera"]["height"]
        self.countdown_duration = config["game"]["countdown_duration"]
        self.result_display_time = config["game"]["result_display_time"]
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # Load Vietnamese-compatible font
        self.font_cache = {}
        self.load_vietnamese_font()

        # Load FPT logo
        self.logo = None
        try:
            self.logo = cv2.imread("asset/LogoFPT.png", cv2.IMREAD_UNCHANGED)
            if self.logo is not None:
                print("✓ FPT Logo loaded successfully")
            else:
                print("⚠ Warning: Could not load FPT logo from asset/LogoFPT.png")
        except Exception as e:
            print(f"⚠ Warning: Error loading logo: {e}")
        
        # Load gesture icons
        self.gesture_icons = {}
        icon_paths = {
            "Búa": "asset/icons/rock-icon.png",
            "Bao": "asset/icons/paper-icon.png",
            "Kéo": "asset/icons/scissors-icon.png"
        }
        for gesture, path in icon_paths.items():
            try:
                icon = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if icon is not None:
                    self.gesture_icons[gesture] = icon
                    print(f"✓ Loaded icon for {gesture}")
                else:
                    print(f"⚠ Warning: Could not load icon from {path}")
            except Exception as e:
                print(f"⚠ Warning: Error loading icon {path}: {e}")
        
        # Load bot-play assets for AI mode
        self.bot_images = {}
        bot_paths = {
            "rule": "asset/bot-play/rule.jpg",
            "rock": "asset/bot-play/rock.jpg",
            "paper": "asset/bot-play/paper.jpg",
            "scissors": "asset/bot-play/sisscors.jpg"
        }
        for key, path in bot_paths.items():
            try:
                img = cv2.imread(path)
                if img is not None:
                    self.bot_images[key] = img
                    print(f"✓ Loaded bot image: {key}")
                else:
                    print(f"⚠ Warning: Could not load bot image from {path}")
            except Exception as e:
                print(f"⚠ Warning: Error loading bot image {path}: {e}")
        
        # AI player state
        self.ai_gesture = None
        self.ai_gesture_time = None

        # AI Strategy settings from config
        ai_config = config["ai_strategy"]
        self.ai_mode = ai_config["default_mode"]  # "random", "cheat", "adaptive"
        self.ai_cheat_threshold = ai_config["cheat_threshold"]
        self.ai_admin_hotkey = ai_config["admin_hotkey"]

        # Auto-balance settings
        auto_balance = ai_config["auto_balance"]
        self.ai_auto_balance_enabled = auto_balance["enabled"]
        self.ai_target_win_rate = auto_balance["target_win_rate"]
        self.ai_check_interval = auto_balance["check_interval"]
        self.ai_min_games = auto_balance["min_games_before_balance"]

        # AI tracking stats
        self.ai_consecutive_losses = 0
        self.ai_player_history = []
        self.ai_total_games = 0
        self.ai_wins = 0

        # UI settings
        self.show_admin_indicator = config["ui"]["show_admin_indicator"]
            
        # GUI window
        self.game_window = None
        
        # Camera
        self.cap = None
        
        # Players
        self.player1 = None
        self.player2 = None
        
        # Game state
        self.game_mode = "play"
        self.countdown_start = None
        self.player1_final = None
        self.player2_final = None
        self.result = ""
        self.result_time = None
        
        # Score tracking
        self.player1_score = 0
        self.player2_score = 0
        self.draws = 0
        
        # Frame timer for consistent FPS
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.target_fps = config["game"]["target_fps"]
        self.timer_interval = int(1000 / self.target_fps)  # milliseconds
        print(f"⏱️  Timer configured: {self.timer_interval}ms interval (target: {self.target_fps} FPS)")
    
    def load_vietnamese_font(self):
        """Load Vietnamese-compatible fonts"""
        font_paths = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/segoeui.ttf"
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    self.font_cache['small'] = ImageFont.truetype(font_path, 24)
                    self.font_cache['medium'] = ImageFont.truetype(font_path, 32)
                    self.font_cache['large'] = ImageFont.truetype(font_path, 60)
                    return
                except:
                    pass
        # Fallback to default
        self.font_cache['small'] = ImageFont.load_default()
        self.font_cache['medium'] = ImageFont.load_default()
        self.font_cache['large'] = ImageFont.load_default()

    def _ai_choose_gesture(self, player_gesture):
        """
        AI chọn gesture - CƠ CẤU mode

        Args:
            player_gesture: Gesture mà player đã ra (đã detect được)

        Returns:
            AI's gesture (counter để thắng, hoặc random)
        """
        win_counters = {"Búa": "Bao", "Bao": "Kéo", "Kéo": "Búa"}
        gestures = ["Búa", "Bao", "Kéo"]

        # Lưu lịch sử player
        if player_gesture:
            self.ai_player_history.append(player_gesture)

        # Nếu không detect được tay player → random
        if player_gesture is None:
            return random.choice(gestures)

        # Check auto-balance first
        if self.ai_auto_balance_enabled:
            self._ai_auto_balance()

        # MODE: "random" - công bằng hoàn toàn
        if self.ai_mode == "random":
            return random.choice(gestures)

        # MODE: "cheat" - luôn thắng (100%)
        if self.ai_mode == "cheat":
            return win_counters[player_gesture]

        # Default: random
        return random.choice(gestures)

    def _ai_auto_balance(self):
        """Tự động điều chỉnh mode để cân bằng tỷ lệ thắng"""
        if self.ai_total_games < self.ai_min_games:
            return  # Chưa đủ games để đánh giá

        if self.ai_total_games % self.ai_check_interval != 0:
            return  # Chỉ check theo interval

        # Tính win rate của AI
        ai_win_rate = self.ai_wins / self.ai_total_games if self.ai_total_games > 0 else 0.5
        old_mode = self.ai_mode

        # Điều chỉnh mode (chỉ random và cheat)
        if ai_win_rate > self.ai_target_win_rate + 0.15:
            # AI thắng quá nhiều → chuyển sang random
            if self.ai_mode != "random":
                self.ai_mode = "random"
                print(f"⚖️ Auto-balance: AI winning too much ({ai_win_rate:.0%}), switching to RANDOM")
        elif ai_win_rate < self.ai_target_win_rate - 0.15:
            # AI thua quá nhiều → chuyển sang cheat
            if self.ai_mode != "cheat":
                self.ai_mode = "cheat"
                print(f"⚖️ Auto-balance: AI losing ({ai_win_rate:.0%}), switching to CHEAT")

        # Update indicator if mode changed
        if old_mode != self.ai_mode and self.game_window:
            self.game_window.update_mode_indicator(self.ai_mode)

    def _ai_update_stats(self, winner):
        """Cập nhật stats sau mỗi trận"""
        self.ai_total_games += 1

        if winner == "p1":  # Player thắng = AI thua
            self.ai_consecutive_losses += 1
        elif winner == "p2":  # AI thắng
            self.ai_consecutive_losses = 0
            self.ai_wins += 1
        # Draw: không đổi gì

    def _ai_toggle_mode(self):
        """Toggle AI mode (cho admin hotkey) - chỉ random và cheat"""
        modes = ["random", "cheat"]
        current_idx = modes.index(self.ai_mode) if self.ai_mode in modes else 0
        next_idx = (current_idx + 1) % len(modes)
        self.ai_mode = modes[next_idx]
        print(f"🎮 AI mode toggled to: {self.ai_mode.upper()}")

        # Update logo color indicator
        if self.game_window:
            self.game_window.update_mode_indicator(self.ai_mode)

    def draw_admin_indicator(self, frame):
        """Draw admin indicator - DISABLED, using logo color instead"""
        # Indicator is now shown via logo color change in UI
        return frame
    
    def draw_text_vietnamese(self, frame, text, position, font_size='medium', color=(255, 255, 255)):
        """Draw Vietnamese text on frame using PIL"""
        # Convert frame to PIL
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Get font
        font = self.font_cache.get(font_size, self.font_cache['medium'])
        
        # Draw text
        draw.text(position, text, font=font, fill=color)
        
        # Convert back to OpenCV
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    def draw_current_prediction(self, frame, gesture, position="top-left"):
        """Draw current prediction text below the last round box with background box and icon"""
        h, w = frame.shape[:2]
        margin = 10
        viz_size = 100  # Reduced to 50% of original (200 -> 100)
        box_width = viz_size
        box_height = 30  # Reduced to 50% (60 -> 30)
        
        # Calculate position below the last round box
        if position == "top-left":
            x_offset = margin
        else:
            x_offset = w - viz_size - margin
        
        y_offset = margin + viz_size + 10  # Below the last round box
        
        # Draw white background box
        cv2.rectangle(frame, 
                     (x_offset, y_offset), 
                     (x_offset + box_width, y_offset + box_height), 
                     (255, 255, 255), -1)  # White filled rectangle
        
        # Draw border
        cv2.rectangle(frame, 
                     (x_offset, y_offset), 
                     (x_offset + box_width, y_offset + box_height), 
                     (0, 255, 0), 2)
        
        # Draw icon if gesture has an icon
        icon_size = 20  # Reduced to 50% (40 -> 20)
        if gesture in self.gesture_icons:
            icon = self.gesture_icons[gesture]
            icon_resized = cv2.resize(icon, (icon_size, icon_size))
            
            icon_x = x_offset + 5
            icon_y = y_offset + (box_height - icon_size) // 2  # Center vertically
            
            # Handle transparency for icon overlay
            if icon_resized.shape[2] == 4:  # Has alpha channel
                alpha = icon_resized[:, :, 3] / 255.0
                roi = frame[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size]
                
                for c in range(3):
                    roi[:, :, c] = (alpha * icon_resized[:, :, c] + (1 - alpha) * roi[:, :, c])
                
                frame[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size] = roi
            else:
                frame[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size] = icon_resized
        
        # Use PIL for Vietnamese text
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        try:
            font_label = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 10)
            font_gesture = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 14)
        except:
            try:
                font_label = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
                font_gesture = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
            except:
                font_label = ImageFont.load_default()
                font_gesture = ImageFont.load_default()
        
        # Draw gesture text next to icon, centered vertically
        text_x = x_offset + icon_size + 8 if gesture in self.gesture_icons else x_offset + 5
        
        # Calculate vertical center for text
        gesture_bbox = draw.textbbox((0, 0), gesture, font=font_gesture)
        text_height = gesture_bbox[3] - gesture_bbox[1]
        text_y = y_offset + (box_height - text_height) // 2
        
        draw.text((text_x, text_y), gesture, font=font_gesture, fill=(0, 150, 0))
        
        # Convert back to OpenCV
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def initialize(self):
        """Initialize camera and players"""
        # Show game window
        self.game_window = self.app_manager.show_game_window()
        
        # Enable fullscreen mode
        self.game_window.showFullScreen()
        
        # Initialize camera with optimized settings
        # STEP 1: Open camera with DirectShow + MJPEG codec
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPEG for stable FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # STEP 2: Lock auto exposure to prevent driver from adjusting when FPS drops
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 0.75 = manual mode (lock exposure)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        
        # STEP 3: Clear buffer by reading a few frames to flush stale data
        for _ in range(5):
            self.cap.read()
        
        print(f"📷 Camera initialized: {self.camera_width}x{self.camera_height} @ {self.target_fps}FPS")
        
        # Create players
        self.player1 = Player(1, self.app_manager.player1_name, self.model, self.scaler)
        self.player2 = Player(2, self.app_manager.player2_name, self.model, self.scaler)
        
        # Start processing threads
        self.player1.start()
        self.player2.start()
        
        # Connect keyboard events
        self.game_window.keyPressEvent = self.handle_key_press

        # Set initial mode indicator (for AI mode)
        if self.game_mode_type == "ai":
            self.game_window.update_mode_indicator(self.ai_mode)

        # Start frame update timer
        self.timer.start(self.timer_interval)
        
        print("🎮 Game initialized successfully!")
        print(f"👤 {self.app_manager.player1_name} vs {self.app_manager.player2_name}")

    def update_frame(self):
        """Update one frame - called by QTimer"""
        frame_start = time.time()
        
        ret, frame = self.cap.read()
        if not ret:
            print("⚠️ Failed to read frame from camera")
            return
        read_time = time.time() - frame_start

        frame = cv2.flip(frame, 1)
        
        # Calculate FPS properly
        self.fps_counter += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_start_time = time.time()
            # Debug timing every second
            print(f"🔍 FPS: {self.current_fps:.1f} | Read: {read_time*1000:.1f}ms")
        
        height, width = frame.shape[:2]
        mid_width = width // 2

        # Split frame into two halves
        frame_left = frame[:, :mid_width]
        frame_right = frame[:, mid_width:]
        
        # Store CLEAN frames for potential capture BEFORE any modification
        clean_frame_left = frame_left.copy()
        clean_frame_right = frame_right.copy()  # Always save clean camera frame

        # AI MODE: Show bot image on right side (AFTER saving clean frame)
        if self.game_mode_type == "ai":
            # Determine which bot image to show
            if self.game_mode == "play" or (self.game_mode == "countdown" and self.ai_gesture is None):
                # Show rule image
                bot_img = self.bot_images.get("rule")
            elif self.ai_gesture:
                # Show AI's choice
                gesture_map = {"Búa": "rock", "Bao": "paper", "Kéo": "scissors"}
                img_key = gesture_map.get(self.ai_gesture, "rule")
                bot_img = self.bot_images.get(img_key)
            else:
                bot_img = self.bot_images.get("rule")
            
            if bot_img is not None:
                # Resize bot image to fit frame_right (only for display)
                bot_img_resized = cv2.resize(bot_img, (frame_right.shape[1], frame_right.shape[0]))
                frame_right = bot_img_resized.copy()
                # Note: clean_frame_right still contains camera frame for capture
        
        # Update frames for each player
        self.player1.update_frame(frame_left, self.game_mode)
        # Always update player2 with camera frame (even in AI mode to keep thread alive)
        # In AI mode, we just won't use player2's prediction
        self.player2.update_frame(clean_frame_right, self.game_mode)

        # Get results from both players (always, to keep tracking active)
        results_p1 = self.player1.get_results()
        results_p2 = self.player2.get_results()
        
        # In AI mode, we'll ignore player2's results but keep the thread running

        # Draw hand landmarks for player 1
        if results_p1['landmarks']:
            self.player1.mp_drawing.draw_landmarks(
                frame_left, results_p1['landmarks'], self.player1.mp_hands.HAND_CONNECTIONS,
                self.player1.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.player1.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
            )

        if self.player1.captured_frame is not None:
            frame_left = self.draw_captured_frame(frame_left, self.player1.captured_frame, 
                                                  "top-left", self.player1.captured_gesture)

        # Draw hand landmarks for player 2 (PvP only)
        if self.game_mode_type == "pvp":
            if results_p2['landmarks']:
                self.player2.mp_drawing.draw_landmarks(
                    frame_right, results_p2['landmarks'], self.player2.mp_hands.HAND_CONNECTIONS,
                    self.player2.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.player2.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )

            if self.player2.captured_frame is not None:
                frame_right = self.draw_captured_frame(frame_right, self.player2.captured_frame, 
                                                       "top-right", self.player2.captured_gesture)

        # Game logic
        if self.game_mode == "play":
            gesture_p1 = results_p1['prediction'] if results_p1['prediction'] else "Không có tay"

            # Draw current prediction below last round box for player 1
            frame_left = self.draw_current_prediction(frame_left, gesture_p1, "top-left")

            # Draw current prediction below last round box for player 2 (PvP only)
            if self.game_mode_type == "pvp":
                gesture_p2 = results_p2['prediction'] if results_p2['prediction'] else "Không có tay"
                frame_right = self.draw_current_prediction(frame_right, gesture_p2, "top-right")
            
            self.game_window.update_status("Sẵn sàng! Nhấn SPACE để bắt đầu", "#00FF00")

        elif self.game_mode == "countdown":
            elapsed = time.time() - self.countdown_start
            remaining = self.countdown_duration - elapsed

            # CAPTURE SINGLE FRAME at t=0 (allows last-second hand changes)
            if remaining <= 0 and not hasattr(self, '_frame_captured'):
                self._captured_frame_left = clean_frame_left.copy()
                self._captured_frame_right = clean_frame_right.copy()
                self._frame_captured = True
                print("📸 Single frame captured at countdown=0 (will process 3x for accuracy)")

            if remaining > 0:
                countdown_text = str(int(remaining) + 1)

                cv2.putText(frame_left, countdown_text, (mid_width//2 - 50, height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
                
                # For AI mode, show countdown on right side only if not showing AI gesture yet
                if self.game_mode_type == "pvp":
                    cv2.putText(frame_right, countdown_text, (mid_width//2 - 50, height//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
                
                # AI MODE: AI will choose AFTER seeing player's gesture (cheat mode)
                
                self.game_window.update_status(f"Chuẩn bị... {countdown_text}", "#00FFFF")
            else:
                # Process captured frame - TRIPLE PROCESSING with VOTING for maximum accuracy
                if not hasattr(self, '_capture_done') and hasattr(self, '_frame_captured'):
                    
                    # CLEAR BUFFERS để không bị contamination
                    self.player1.clear_tracking_buffer()
                    if self.game_mode_type == "pvp":
                        self.player2.clear_tracking_buffer()
                    
                    # Process SAME FRAME 3 TIMES with different thresholds - highest accuracy!
                    print("🎯 Processing single frame 3x with different thresholds for maximum accuracy...")
                    pred_p1, landmarks_p1 = self.player1.process_single_frame_triple(self._captured_frame_left)

                    if self.game_mode_type == "ai":
                        # AI "CƠ CẤU": Chọn gesture SAU KHI biết player ra gì
                        self.ai_gesture = self._ai_choose_gesture(pred_p1)
                        self.ai_gesture_time = time.time()
                        print(f"🤖 AI counter-picked: {self.ai_gesture} (vs player's {pred_p1})")
                        pred_p2 = self.ai_gesture
                        landmarks_p2 = True  # Fake for AI
                    else:
                        pred_p2, landmarks_p2 = self.player2.process_single_frame_triple(self._captured_frame_right)
                    
                    # Store results immediately
                    self._pred_p1 = pred_p1
                    self._pred_p2 = pred_p2
                    self._landmarks_p1 = landmarks_p1
                    self._landmarks_p2 = landmarks_p2
                    
                    self._capture_time = time.time()
                    self._capture_done = True
                    self._predictions_ready = True  # Already have predictions!
                
                # Check if we have predictions ready
                if hasattr(self, '_predictions_ready') and self._predictions_ready:
                    processing_time = time.time() - self._capture_time
                    
                    # Use isolated predictions (already processed)
                    pred_p1 = self._pred_p1
                    pred_p2 = self._pred_p2
                    landmarks_p1 = self._landmarks_p1
                    landmarks_p2 = self._landmarks_p2
                    
                    # If detection failed, show message briefly
                    if pred_p1 is None or pred_p2 is None:
                        if processing_time < 1.0:
                            if pred_p1 is None:
                                cv2.putText(frame_left, "NO HAND DETECTED", (50, height//2),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                            if pred_p2 is None and self.game_mode_type == "pvp":
                                cv2.putText(frame_right, "NO HAND DETECTED", (50, height//2),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                            self.game_window.update_status("Không phát hiện tay!", "#FF0000")
                            return
                        
                    # Store results with landmarks
                    captured_with_landmarks_left = self._captured_frame_left.copy()
                    
                    if landmarks_p1:
                        self.player1.mp_drawing.draw_landmarks(
                            captured_with_landmarks_left, landmarks_p1, self.player1.mp_hands.HAND_CONNECTIONS,
                            self.player1.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            self.player1.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                        )
                    
                    self.player1.captured_frame = self._captured_frame_left
                    self.player1.captured_frame_with_landmarks = captured_with_landmarks_left
                    self.player1.captured_gesture = pred_p1 if pred_p1 else "Không có tay"
                    
                    # Handle player 2 based on mode
                    if self.game_mode_type == "pvp":
                        captured_with_landmarks_right = self._captured_frame_right.copy()
                        if landmarks_p2:
                            self.player2.mp_drawing.draw_landmarks(
                                captured_with_landmarks_right, landmarks_p2, self.player2.mp_hands.HAND_CONNECTIONS,
                                self.player2.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                self.player2.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                            )
                        self.player2.captured_frame = self._captured_frame_right
                        self.player2.captured_frame_with_landmarks = captured_with_landmarks_right
                        self.player2.captured_gesture = pred_p2 if pred_p2 else "Không có tay"
                    else:
                        # AI mode - no captured frame for AI
                        self.player2.captured_gesture = pred_p2 if pred_p2 else "Không có tay"

                    self.player1_final = pred_p1
                    self.player2_final = pred_p2
                    winner = determine_winner(self.player1_final, self.player2_final)

                    # Update AI stats for adaptive mode
                    if self.game_mode_type == "ai":
                        self._ai_update_stats(winner)

                    if winner == "p1":
                        self.player1_score += 1
                        self.result = f"{self.app_manager.player1_name} Thắng!"
                        self.audio_manager.play_winner_sound("asset/result/player-1.mp3")
                    elif winner == "p2":
                        self.player2_score += 1
                        self.result = f"{self.app_manager.player2_name} Thắng!"
                        self.audio_manager.play_winner_sound("asset/result/player-2.mp3")
                    elif winner == "draw":
                        self.draws += 1
                        self.result = "Hòa!"
                        self.audio_manager.play_winner_sound("asset/result/tie.wav")
                    else:
                        self.result = "Không phát hiện tay!"

                    self.result_time = time.time()
                    self.game_mode = "result"
                    
                    # Update GUI scores
                    self.game_window.update_scores(self.player1_score, self.player2_score, self.draws)
                    
                    delattr(self, '_capture_done')
                    delattr(self, '_predictions_ready')

        elif self.game_mode == "result":
            # Draw current prediction below last round box for player 1
            gesture_text_p1 = self.player1_final if self.player1_final else "Không có tay"
            frame_left = self.draw_current_prediction(frame_left, gesture_text_p1, "top-left")

            # Draw current prediction below last round box for player 2
            gesture_text_p2 = self.player2_final if self.player2_final else "Không có tay"
            frame_right = self.draw_current_prediction(frame_right, gesture_text_p2, "top-right")
            
            # Determine result color
            if self.app_manager.player1_name in self.result:
                result_color = "#00FFFF"
            elif self.app_manager.player2_name in self.result:
                result_color = "#FFA500"
            else:
                result_color = "#00FF00"
                
            self.game_window.update_status(self.result, result_color)

            if time.time() - self.result_time > 3:
                self.game_mode = "play"
                self.player1_final = None
                self.player2_final = None
                self.result = ""

        # Combine frames
        combined_frame = np.hstack([frame_left, frame_right])

        # Draw center line
        cv2.line(combined_frame, (mid_width, 0), (mid_width, height), (255, 255, 255), 2)
        
        # Draw FPS counter at top-left corner (small size)
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(combined_frame, fps_text, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Draw logo at center top (reduced size)
        combined_frame = self.draw_logo(combined_frame, self.logo, position="top-center", max_height=40)

        # Display result text on video
        if self.result:
            # Use PIL for Vietnamese text rendering
            from PIL import Image, ImageDraw, ImageFont
            import os
            
            # Convert frame to PIL Image
            pil_img = Image.fromarray(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # Try to use a Vietnamese-compatible font
            try:
                # Try system fonts that support Vietnamese
                font_size = 60
                font_paths = [
                    "C:/Windows/Fonts/arial.ttf",
                    "C:/Windows/Fonts/calibri.ttf",
                    "C:/Windows/Fonts/segoeui.ttf"
                ]
                font = None
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, font_size)
                        break
                if font is None:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # Get text bounding box
            bbox = draw.textbbox((0, 0), self.result, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            text_x = (width - text_width) // 2
            text_y = height - 80
            
            # Draw background rectangle
            padding = 20
            draw.rectangle(
                [(text_x - padding, text_y - padding),
                 (text_x + text_width + padding, text_y + text_height + padding)],
                fill=(0, 0, 0, 180)
            )
            
            # Determine color
            if self.app_manager.player1_name in self.result:
                color = (0, 255, 255)
            elif self.app_manager.player2_name in self.result:
                color = (255, 165, 0)
            else:
                color = (0, 255, 0)
            
            # Draw text
            draw.text((text_x, text_y), self.result, font=font, fill=color)
            
            # Convert back to OpenCV format
            combined_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Draw admin indicator for AI mode
        combined_frame = self.draw_admin_indicator(combined_frame)

        # Update GUI with frame
        before_gui = time.time()
        self.game_window.update_frame(combined_frame)
        gui_time = time.time() - before_gui
        
        total_time = time.time() - frame_start
        # Print timing once per second
        if hasattr(self, '_last_debug_time') and time.time() - self._last_debug_time >= 1.0:
            print(f"⏱️  Frame timing: Total={total_time*1000:.1f}ms | GUI={gui_time*1000:.1f}ms")
            self._last_debug_time = time.time()
        if not hasattr(self, '_last_debug_time'):
            self._last_debug_time = time.time()

    def draw_captured_frame(self, frame, captured_frame, position="top-left", gesture_text=""):
        """Draw captured frame from previous round with icon"""
        if captured_frame is None:
            return frame

        viz_size = 100  # Reduced to 50% of original (200 -> 100)
        h, w = frame.shape[:2]
        margin = 10

        if position == "top-left":
            x_offset, y_offset = margin, margin
        else:
            x_offset, y_offset = w - viz_size - margin, margin

        viz_img = cv2.resize(captured_frame, (viz_size, viz_size))

        # Draw white rectangle at bottom for text (smaller area)
        text_area_height = 25  # Smaller text area
        cv2.rectangle(viz_img, (0, viz_size - text_area_height), (viz_size, viz_size), (255, 255, 255), -1)

        if gesture_text:
            # Draw icon if gesture has one (smaller)
            icon_size = 18  # Much smaller to not cover text
            if gesture_text in self.gesture_icons:
                icon = self.gesture_icons[gesture_text]
                icon_resized = cv2.resize(icon, (icon_size, icon_size))
                
                icon_x = 3
                icon_y = viz_size - text_area_height + 3  # Position at top of text area
                
                # Handle transparency for icon overlay
                if icon_resized.shape[2] == 4:  # Has alpha channel
                    alpha = icon_resized[:, :, 3] / 255.0
                    roi = viz_img[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size]
                    
                    for c in range(3):
                        roi[:, :, c] = (alpha * icon_resized[:, :, c] + (1 - alpha) * roi[:, :, c])
                    
                    viz_img[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size] = roi
                else:
                    viz_img[icon_y:icon_y+icon_size, icon_x:icon_x+icon_size] = icon_resized
            
            # Use PIL for Vietnamese text
            pil_img = Image.fromarray(cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            try:
                font_gesture = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 12)
            except:
                font_gesture = ImageFont.load_default()
            
            # Draw gesture text next to icon, centered vertically in text area
            text_x = icon_size + 6 if gesture_text in self.gesture_icons else 3
            text_y = viz_size - text_area_height + 5
            draw.text((text_x, text_y), gesture_text, font=font_gesture, fill=(0, 120, 0))
            
            viz_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        cv2.rectangle(frame, (x_offset-2, y_offset-2),
                      (x_offset+viz_size+2, y_offset+viz_size+2),
                      (255, 255, 255), 2)

        frame[y_offset:y_offset+viz_size, x_offset:x_offset+viz_size] = viz_img

        return frame

    def draw_logo(self, frame, logo_img, position="top-center", max_height=80):
        """Draw FPT logo on frame with rounded white background box"""
        if logo_img is None:
            return frame

        h, w = frame.shape[:2]

        logo_h, logo_w = logo_img.shape[:2]
        aspect_ratio = logo_w / logo_h
        new_height = max_height
        new_width = int(new_height * aspect_ratio)

        logo_resized = cv2.resize(logo_img, (new_width, new_height))

        # Box dimensions with padding (reduced)
        padding = 6  # Reduced to 50% (12 -> 6)
        box_width = new_width + (padding * 2)
        box_height = new_height + (padding * 2)
        radius = 8  # Reduced to ~50% (15 -> 8)

        if position == "top-center":
            box_x = (w - box_width) // 2
            box_y = 10
        elif position == "top-right":
            box_x = w - box_width - 10
            box_y = 10
        else:
            box_x = 10
            box_y = 10

        # Create a white rounded rectangle using PIL for better quality
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Draw rounded rectangle (white background)
        draw.rounded_rectangle(
            [(box_x, box_y), (box_x + box_width, box_y + box_height)],
            radius=radius,
            fill=(255, 255, 255),
            outline=(200, 200, 200),
            width=2
        )
        
        # Convert back to OpenCV
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Calculate logo position (centered in box)
        x_offset = box_x + padding
        y_offset = box_y + padding

        # Draw logo on top of the white box
        if logo_resized.shape[2] == 4:
            alpha = logo_resized[:, :, 3] / 255.0
            roi = frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width]

            for c in range(3):
                roi[:, :, c] = (alpha * logo_resized[:, :, c] +
                               (1 - alpha) * roi[:, :, c])

            frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = roi
        else:
            frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = logo_resized

        return frame

    def handle_key_press(self, event):
        """Handle keyboard input"""
        key = event.key()
        
        if key == Qt.Key_Q:
            self.cleanup()
            QApplication.quit()
        elif key == Qt.Key_Space and self.game_mode == "play":
            self.game_mode = "countdown"
            self.countdown_start = time.time()
            self.result = ""
            
            # Reset capture flags for new round
            for attr in ['_frame_captured', '_capture_done', '_predictions_ready', 
                        '_pred_p1', '_pred_p2', '_landmarks_p1', '_landmarks_p2', 
                        '_capture_time', '_captured_frame_left', '_captured_frame_right']:
                if hasattr(self, attr):
                    delattr(self, attr)
            
            # Reset AI gesture for new round
            self.ai_gesture = None
            self.ai_gesture_time = None
            # Play countdown sound and fade background music
            self.audio_manager.play_countdown_sound()
        elif key == Qt.Key_R:
            # Reset scores only
            self.reset_scores()
        elif key == Qt.Key_Escape:
            # Return to game mode selection
            self.restart_game()
        elif key == Qt.Key_F11:
            # Toggle fullscreen
            if self.game_window.isFullScreen():
                self.game_window.showNormal()
            else:
                self.game_window.showFullScreen()
        elif key == Qt.Key_C and self.game_mode_type == "ai":
            # Admin hotkey: Toggle AI mode (C key) - subtle, no status message
            self._ai_toggle_mode()

    def reset_scores(self):
        """Reset scores only"""
        self.game_mode = "play"
        self.player1_final = None
        self.player2_final = None
        self.result = ""
        self.player1_score = 0
        self.player2_score = 0
        self.draws = 0

        # Reset AI stats
        self.ai_consecutive_losses = 0
        self.ai_player_history = []
        self.ai_total_games = 0
        self.ai_wins = 0
        # Reset AI mode to default from config
        self.ai_mode = self.config["ai_strategy"]["default_mode"]

        if self.game_window:
            self.game_window.update_scores(0, 0, 0)
            self.game_window.update_status("Điểm đã được reset!", "#FFD700")
    
    def restart_game(self):
        """Restart game with new names"""
        # Stop current game
        self.timer.stop()
        
        # Hide game window
        if self.game_window:
            self.game_window.hide()
        
        # Reset scores
        self.reset_scores()
        
        # Show name dialog to re-enter names
        self.app_manager.show_name_dialog_for_restart(self)

    def cleanup(self):
        """Cleanup resources"""
        print("\n🛑 Cleaning up...")
        
        # Stop timer
        if self.timer:
            self.timer.stop()
        
        # Stop players
        if self.player1:
            self.player1.stop()
        if self.player2:
            self.player2.stop()
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        # Stop all audio
        self.audio_manager.stop_all()
        
        # Print final scores
        print("\n🏆 FINAL SCORES")
        print(f"{self.app_manager.player1_name}: {self.player1_score} | {self.app_manager.player2_name}: {self.player2_score} | Draws: {self.draws}")
        if self.player1_score > self.player2_score:
            print(f"{self.app_manager.player1_name} WINS! 🎉")
        elif self.player2_score > self.player1_score:
            print(f"{self.app_manager.player2_name} WINS! 🎉")
        else:
            print("It's a TIE! 🤝")


# =====================================
# Main entry point
# =====================================
def main():
    """Main function to run the GUI game"""
    # Load configuration from file
    config = load_config("config.json")

    print("🎮 Rock Paper Scissors - PyQt5 GUI Version")
    print("=" * 50)
    print(f"📋 Config loaded: Camera {config['camera']['width']}x{config['camera']['height']}")
    print(f"🤖 AI default mode: {config['ai_strategy']['default_mode']}")

    # Create application
    app_manager = RPSApplication()

    # Create game instance
    game = None

    def on_loading_complete():
        """Initialize game after loading"""
        nonlocal game
        game = RPSGameGUI(app_manager, config)
        game.initialize()

    # Connect loading complete signal
    app_manager.on_loading_complete = on_loading_complete

    # Start application
    app_manager.start()

    # Run event loop
    exit_code = app_manager.exec()

    # Cleanup
    if game:
        game.cleanup()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
