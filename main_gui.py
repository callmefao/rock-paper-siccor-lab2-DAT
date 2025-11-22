# =====================================
# Rock Paper Scissors - PyQt5 GUI Version
# =====================================
import cv2
import numpy as np
import sys
import time
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication
from joblib import load
from PIL import Image, ImageDraw, ImageFont
import os

# Import UI components
from ui_main import RPSApplication, GameWindow

# Import game components
from hand_feature_extractor import HandFeatureExtractor
from main import Player, normalize_hand_orientation, determine_winner, play_sound


# =====================================
# Enhanced RPSGame with PyQt5 Integration
# =====================================
class RPSGameGUI:
    """Main game class with PyQt5 GUI integration"""

    def __init__(self, app_manager, model_path, scaler_path, camera_width=1280, camera_height=720, countdown_duration=3):
        """
        Initialize game with model, scaler, and GUI

        Args:
            app_manager: RPSApplication instance
            model_path: Path to trained model
            scaler_path: Path to feature scaler
            camera_width: Width of camera capture
            camera_height: Height of camera capture
            countdown_duration: Duration of countdown in seconds
        """
        self.app_manager = app_manager
        self.audio_manager = app_manager.audio_manager  # Store audio manager reference
        self.game_mode = app_manager.game_mode  # "single" or "two"
        
        # AI bot images
        self.bot_images = {}
        if self.game_mode == "single":
            try:
                self.bot_images['rule'] = cv2.imread("asset/bot-play/rule.jpg")
                self.bot_images['rock'] = cv2.imread("asset/bot-play/rock.jpg")
                self.bot_images['paper'] = cv2.imread("asset/bot-play/paper.jpg")
                self.bot_images['scissors'] = cv2.imread("asset/bot-play/sisscors.jpg")
                print("✓ Bot images loaded successfully")
            except Exception as e:
                print(f"⚠ Warning: Could not load bot images: {e}")
        
        # AI bot gesture (for single player mode)
        self.bot_gesture = None
        
        # Load trained model and scaler
        self.model = load(model_path)
        self.scaler = load(scaler_path)

        # Game configuration
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.countdown_duration = countdown_duration

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
            "Giấy": "asset/icons/paper-icon.png",
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
            
        # GUI window
        self.game_window = None
        
        # Camera
        self.cap = None
        
        # Players
        self.player1 = None
        self.player2 = None
        
        # Game state
        self.game_mode_state = "play"  # Renamed to avoid conflict with self.game_mode (single/two)
        self.countdown_start = None
        self.player1_final = None
        self.player2_final = None
        self.result = ""
        self.result_time = None
        
        # Capture state (thread-safe)
        self.capture_state = {
            'done': False,
            'predictions_ready': False,
            'capture_time': None,
            'captured_frames': {'left': None, 'right': None}
        }
        
        # Score tracking
        self.player1_score = 0
        self.player2_score = 0
        self.draws = 0
        
        # Frame timer for consistent FPS
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.target_fps = 30
        self.timer_interval = int(1000 / self.target_fps)  # milliseconds
    
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
        viz_size = 280  # Matches the captured frame box size
        box_width = viz_size
        box_height = 70
        
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
        icon_size = 50
        if gesture in self.gesture_icons:
            icon = self.gesture_icons[gesture]
            icon_resized = cv2.resize(icon, (icon_size, icon_size))
            
            icon_x = x_offset + 10
            icon_y = y_offset + 10
            
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
            font_label = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
            font_gesture = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 22)  # Slightly smaller
        except:
            try:
                font_label = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
                font_gesture = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
            except:
                font_label = ImageFont.load_default()
                font_gesture = ImageFont.load_default()
        
        # Calculate centered text position for label
        label_bbox = draw.textbbox((0, 0), "Hiện tại:", font=font_label)
        label_width = label_bbox[2] - label_bbox[0]
        label_x = x_offset + (box_width - label_width) // 2
        
        # Draw centered label with dark text for white background
        draw.text((label_x, y_offset + 8), "Hiện tại:", font=font_label, fill=(0, 0, 0))
        
        # Draw gesture text next to icon
        text_x = x_offset + icon_size + 20 if gesture in self.gesture_icons else x_offset + 8
        draw.text((text_x, y_offset + 32), gesture, font=font_gesture, fill=(0, 150, 0))
        
        # Convert back to OpenCV
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def initialize(self):
        """Initialize camera and players"""
        # Show game window
        self.game_window = self.app_manager.show_game_window()
        
        # Enable fullscreen mode
        self.game_window.showFullScreen()
        
        # Initialize camera with optimized settings
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        
        print(f"📷 Camera initialized: {self.camera_width}x{self.camera_height} @ {self.target_fps}FPS")
        
        # Create players
        self.player1 = Player(1, self.app_manager.player1_name, self.model, self.scaler)
        
        if self.game_mode == "two":
            self.player2 = Player(2, self.app_manager.player2_name, self.model, self.scaler)
            self.player2.start()
        else:
            self.player2 = None  # No player 2 in single player mode
        
        # Start processing threads
        self.player1.start()
        
        # Connect keyboard events
        self.game_window.keyPressEvent = self.handle_key_press
        
        # Start frame update timer
        self.timer.start(self.timer_interval)
        
        print("🎮 Game initialized successfully!")
        print(f"👤 {self.app_manager.player1_name} vs {self.app_manager.player2_name}")

    def update_frame(self):
        """Update one frame - called by QTimer"""
        ret, frame = self.cap.read()
        if not ret:
            print("⚠️ Failed to read frame from camera")
            return

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        mid_width = width // 2

        # Split frame into two halves
        frame_left = frame[:, :mid_width]
        frame_right = frame[:, mid_width:]
        
        # Store CLEAN frames for potential capture
        clean_frame_left = frame_left.copy()
        clean_frame_right = frame_right.copy()

        # Update frames for player 1
        self.player1.update_frame(frame_left, self.game_mode)
        results_p1 = self.player1.get_results()

        # For two-player mode, process player 2
        if self.game_mode == "two":
            self.player2.update_frame(frame_right, self.game_mode)
            results_p2 = self.player2.get_results()
        else:
            # Single player mode - no processing for right side
            results_p2 = {'features': None, 'landmarks': None, 'prediction': None}

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

        # Handle right side based on game mode
        if self.game_mode == "two":
            # Two-player mode: draw landmarks and captured frame for player 2
            if results_p2['landmarks']:
                self.player2.mp_drawing.draw_landmarks(
                    frame_right, results_p2['landmarks'], self.player2.mp_hands.HAND_CONNECTIONS,
                    self.player2.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.player2.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )

            if self.player2.captured_frame is not None:
                frame_right = self.draw_captured_frame(frame_right, self.player2.captured_frame, 
                                                       "top-right", self.player2.captured_gesture)
        else:
            # Single-player mode: show bot image on right side
            if self.game_mode_state == "play":
                # Show rule image during play mode
                if 'rule' in self.bot_images and self.bot_images['rule'] is not None:
                    bot_img = cv2.resize(self.bot_images['rule'], (frame_right.shape[1], frame_right.shape[0]))
                    frame_right = bot_img
            elif self.game_mode_state == "countdown":
                # During countdown, check if we're at 1
                elapsed = time.time() - self.countdown_start
                remaining = self.countdown_duration - elapsed
                if int(remaining) + 1 == 1 and self.bot_gesture:
                    # Show bot gesture image at countdown = 1
                    gesture_map = {"Búa": "rock", "Giấy": "paper", "Kéo": "scissors"}
                    img_key = gesture_map.get(self.bot_gesture)
                    if img_key and img_key in self.bot_images and self.bot_images[img_key] is not None:
                        bot_img = cv2.resize(self.bot_images[img_key], (frame_right.shape[1], frame_right.shape[0]))
                        frame_right = bot_img
                else:
                    # Show rule image before countdown reaches 1
                    if 'rule' in self.bot_images and self.bot_images['rule'] is not None:
                        bot_img = cv2.resize(self.bot_images['rule'], (frame_right.shape[1], frame_right.shape[0]))
                        frame_right = bot_img
            elif self.game_mode_state == "result" and self.bot_gesture:
                # Show bot gesture during result
                gesture_map = {"Búa": "rock", "Giấy": "paper", "Kéo": "scissors"}
                img_key = gesture_map.get(self.bot_gesture)
                if img_key and img_key in self.bot_images and self.bot_images[img_key] is not None:
                    bot_img = cv2.resize(self.bot_images[img_key], (frame_right.shape[1], frame_right.shape[0]))
                    frame_right = bot_img

        # Game logic
        if self.game_mode_state == "play":
            gesture_p1 = results_p1['prediction'] if results_p1['prediction'] else "Không có tay"

            # Draw current prediction below last round box for player 1
            frame_left = self.draw_current_prediction(frame_left, gesture_p1, "top-left")

            if self.game_mode == "two":
                gesture_p2 = results_p2['prediction'] if results_p2['prediction'] else "Không có tay"
                # Draw current prediction below last round box for player 2
                frame_right = self.draw_current_prediction(frame_right, gesture_p2, "top-right")
            
            self.game_window.update_status("Sẵn sàng! Nhấn SPACE để bắt đầu", "#00FF00")

        elif self.game_mode_state == "countdown":
            elapsed = time.time() - self.countdown_start
            remaining = self.countdown_duration - elapsed

            if remaining > 0:
                countdown_text = str(int(remaining) + 1)
                
                # Check if countdown reached 1
                if int(remaining) + 1 == 1:
                    # Random AI gesture at countdown = 1
                    if self.game_mode == "single" and self.bot_gesture is None:
                        import random
                        self.bot_gesture = random.choice(["Búa", "Giấy", "Kéo"])
                        print(f"🤖 Bot chose: {self.bot_gesture}")

                cv2.putText(frame_left, countdown_text, (mid_width//2 - 50, height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
                
                if self.game_mode == "two":
                    cv2.putText(frame_right, countdown_text, (mid_width//2 - 50, height//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
                else:
                    # Single player mode: show countdown on bot side too
                    cv2.putText(frame_right, countdown_text, (mid_width//2 - 50, height//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
                
                self.game_window.update_status(f"Chuẩn bị... {countdown_text}", "#00FFFF")
            else:
                # Capture and process logic (same as original)
                if not self.capture_state['done']:
                    self.capture_state['captured_frames']['left'] = clean_frame_left.copy()
                    
                    with self.player1.lock:
                        self.player1.prediction_buffer.clear()
                    
                    self.player1.update_frame(self.capture_state['captured_frames']['left'], self.game_mode_state)
                    
                    if self.game_mode == "two":
                        self.capture_state['captured_frames']['right'] = clean_frame_right.copy()
                        with self.player2.lock:
                            self.player2.prediction_buffer.clear()
                        self.player2.update_frame(self.capture_state['captured_frames']['right'], self.game_mode_state)
                    
                    self.capture_state['capture_time'] = time.time()
                    self.capture_state['done'] = True
                    self.capture_state['predictions_ready'] = False
                
                processing_time = time.time() - self.capture_state['capture_time']
                
                if processing_time < 0.3:
                    cv2.putText(frame_left, "PROCESSING...", (mid_width//2 - 200, height//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 6)
                    cv2.putText(frame_right, "PROCESSING...", (mid_width//2 - 200, height//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 6)
                    
                    self.game_window.update_status("Đang xử lý...", "#00FFFF")
                    
                    self.player1.update_frame(self.capture_state['captured_frames']['left'], self.game_mode_state)
                    if self.game_mode == "two":
                        self.player2.update_frame(self.capture_state['captured_frames']['right'], self.game_mode_state)
                else:
                    if not self.capture_state['predictions_ready']:
                        results_p1 = self.player1.get_results()
                        
                        pred_p1 = results_p1['prediction']
                        landmarks_p1 = results_p1['landmarks']
                        
                        if self.game_mode == "two":
                            results_p2 = self.player2.get_results()
                            pred_p2 = results_p2['prediction']
                            landmarks_p2 = results_p2['landmarks']
                        else:
                            # Single player mode: use bot gesture
                            pred_p2 = self.bot_gesture
                            landmarks_p2 = "bot"  # Dummy value for bot
                        
                        # Check if predictions are ready
                        if self.game_mode == "two":
                            if (pred_p1 is None or pred_p2 is None or 
                                landmarks_p1 is None or landmarks_p2 is None):
                                if processing_time < 2.0:
                                    if pred_p1 is None or landmarks_p1 is None:
                                        cv2.putText(frame_left, "DETECTING...", (50, height//2),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)
                                    if pred_p2 is None or landmarks_p2 is None:
                                        cv2.putText(frame_right, "DETECTING...", (50, height//2),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)
                                    
                                    self.game_window.update_status("Đang phát hiện...", "#FFA500")
                                    
                                    self.player1.update_frame(self.capture_state['captured_frames']['left'], self.game_mode_state)
                                    self.player2.update_frame(self.capture_state['captured_frames']['right'], self.game_mode_state)
                                    return
                                else:
                                    pred_p1 = pred_p1 if pred_p1 else None
                                    pred_p2 = pred_p2 if pred_p2 else None
                        else:
                            # Single player mode: only check player 1
                            if pred_p1 is None or landmarks_p1 is None:
                                if processing_time < 2.0:
                                    cv2.putText(frame_left, "DETECTING...", (50, height//2),
                                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)
                                    
                                    self.game_window.update_status("Đang phát hiện...", "#FFA500")
                                    
                                    self.player1.update_frame(self.capture_state['captured_frames']['left'], self.game_mode_state)
                                    return
                                else:
                                    pred_p1 = pred_p1 if pred_p1 else None
                        
                        # Store results
                        captured_with_landmarks_left = self.capture_state['captured_frames']['left'].copy()
                        
                        if landmarks_p1:
                            self.player1.mp_drawing.draw_landmarks(
                                captured_with_landmarks_left, landmarks_p1, self.player1.mp_hands.HAND_CONNECTIONS,
                                self.player1.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                self.player1.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                            )
                        
                        self.player1.captured_frame = self.capture_state['captured_frames']['left']
                        self.player1.captured_frame_with_landmarks = captured_with_landmarks_left
                        self.player1.captured_gesture = pred_p1 if pred_p1 else "Không có tay"
                        
                        if self.game_mode == "two":
                            captured_with_landmarks_right = self.capture_state['captured_frames']['right'].copy()
                            
                            if landmarks_p2:
                                self.player2.mp_drawing.draw_landmarks(
                                    captured_with_landmarks_right, landmarks_p2, self.player2.mp_hands.HAND_CONNECTIONS,
                                    self.player2.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                    self.player2.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                                )
                            
                            self.player2.captured_frame = self.capture_state['captured_frames']['right']
                            self.player2.captured_frame_with_landmarks = captured_with_landmarks_right
                            self.player2.captured_gesture = pred_p2 if pred_p2 else "Không có tay"

                        self.player1_final = pred_p1
                        self.player2_final = pred_p2
                        winner = determine_winner(self.player1_final, self.player2_final)

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
                        self.game_mode_state = "result"
                        
                        # Update GUI scores
                        self.game_window.update_scores(self.player1_score, self.player2_score, self.draws)
                        
                        # Reset capture state for next round
                        self.capture_state['done'] = False
                        self.capture_state['predictions_ready'] = False
                        
                        # Reset bot gesture for next round
                        if self.game_mode == "single":
                            self.bot_gesture = None

        elif self.game_mode_state == "result":
            # Draw current prediction below last round box for player 1
            gesture_text_p1 = self.player1_final if self.player1_final else "Không có tay"
            frame_left = self.draw_current_prediction(frame_left, gesture_text_p1, "top-left")

            if self.game_mode == "two":
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
                self.game_mode_state = "play"
                self.player1_final = None
                self.player2_final = None
                self.result = ""

        # Combine frames
        combined_frame = np.hstack([frame_left, frame_right])

        # Draw center line
        cv2.line(combined_frame, (mid_width, 0), (mid_width, height), (255, 255, 255), 2)

        # Draw logo at center top (replacing FPT RPS text)
        combined_frame = self.draw_logo(combined_frame, self.logo, position="top-center", max_height=80)

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

        # Update GUI with frame
        self.game_window.update_frame(combined_frame)

    def draw_captured_frame(self, frame, captured_frame, position="top-left", gesture_text=""):
        """Draw captured frame from previous round with icon"""
        if captured_frame is None:
            return frame

        viz_size = 280  # Increased by 50% from original 187 (187 * 1.5 = 280)
        h, w = frame.shape[:2]
        margin = 10

        if position == "top-left":
            x_offset, y_offset = margin, margin
        else:
            x_offset, y_offset = w - viz_size - margin, margin

        viz_img = cv2.resize(captured_frame, (viz_size, viz_size))

        # Draw white rectangle at bottom for text
        cv2.rectangle(viz_img, (0, viz_size - 60), (viz_size, viz_size), (255, 255, 255), -1)

        if gesture_text:
            # Draw icon if gesture has one
            icon_size = 45  # Increased for larger box
            if gesture_text in self.gesture_icons:
                icon = self.gesture_icons[gesture_text]
                icon_resized = cv2.resize(icon, (icon_size, icon_size))
                
                icon_x = 8
                icon_y = viz_size - 52
                
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
                font_small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 16)
                font_medium = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
            except:
                font_small = ImageFont.load_default()
                font_medium = ImageFont.load_default()
            
            # Calculate centered text position for label
            label_bbox = draw.textbbox((0, 0), "Lượt trước:", font=font_small)
            label_width = label_bbox[2] - label_bbox[0]
            label_x = (viz_size - label_width) // 2
            
            draw.text((label_x, viz_size - 54), "Lượt trước:", font=font_small, fill=(0, 0, 0))
            
            # Draw gesture text next to icon
            text_x = icon_size + 15 if gesture_text in self.gesture_icons else 8
            draw.text((text_x, viz_size - 28), gesture_text, font=font_medium, fill=(0, 150, 0))
            
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

        # Box dimensions with padding
        padding = 12
        box_width = new_width + (padding * 2)
        box_height = new_height + (padding * 2)
        radius = 15  # Rounded corner radius

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
        elif key == Qt.Key_Space and self.game_mode_state == "play":
            self.game_mode_state = "countdown"
            self.countdown_start = time.time()
            self.result = ""
            # Play countdown sound and fade background music
            self.audio_manager.play_countdown_sound()
        elif key == Qt.Key_R:
            # Reset scores only
            self.reset_scores()
        elif key == Qt.Key_Escape:
            # Return to main menu (mode selection)
            self.return_to_menu()
        elif key == Qt.Key_F11:
            # Toggle fullscreen
            if self.game_window.isFullScreen():
                self.game_window.showNormal()
            else:
                self.game_window.showFullScreen()

    def reset_scores(self):
        """Reset scores only"""
        self.game_mode_state = "play"
        self.player1_final = None
        self.player2_final = None
        self.result = ""
        self.player1_score = 0
        self.player2_score = 0
        self.draws = 0
        
        if self.game_window:
            self.game_window.update_scores(0, 0, 0)
            self.game_window.update_status("Điểm đã được reset!", "#FFD700")
    
    def return_to_menu(self):
        """Return to main menu (mode selection)"""
        # Stop current game
        self.timer.stop()
        
        # Hide game window
        if self.game_window:
            self.game_window.hide()
        
        # Reset scores
        self.reset_scores()
        
        # Clean up players
        if self.player1:
            self.player1.stop()
        if self.player2 and self.game_mode == "two":
            self.player2.stop()
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        # Show mode selection dialog
        self.app_manager.show_mode_dialog()

    def cleanup(self):
        """Cleanup resources"""
        print("\n🛑 Cleaning up...")
        
        # Stop timer
        if self.timer:
            self.timer.stop()
        
        # Stop players
        if self.player1:
            self.player1.stop()
        if self.player2 and self.game_mode == "two":
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
    # Configuration
    MODEL_PATH = "model/rps_ridge_model.joblib"
    SCALER_PATH = "model/rps_scaler.joblib"
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    COUNTDOWN_DURATION = 3
    
    print("🎮 Rock Paper Scissors - PyQt5 GUI Version")
    print("=" * 50)
    
    # Create application
    app_manager = RPSApplication()
    
    # Create game instance
    game = None
    
    def on_loading_complete():
        """Initialize game after loading"""
        nonlocal game
        game = RPSGameGUI(app_manager, MODEL_PATH, SCALER_PATH, 
                         CAMERA_WIDTH, CAMERA_HEIGHT, COUNTDOWN_DURATION)
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
