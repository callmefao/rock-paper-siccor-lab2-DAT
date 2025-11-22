# =====================================
# UI Components for Rock Paper Scissors Game
# =====================================
import sys
import time
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QFrame, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor
import cv2
import numpy as np
from gtts import gTTS
import threading
import winsound
import pygame
import os


# =====================================
# Audio Manager
# =====================================
class AudioManager:
    """Manages background music and sound effects with volume control (pygame + winsound)"""
    
    def __init__(self):
        # Initialize pygame mixer
        pygame.mixer.init()
        self.bg_music_playing = False
        self.normal_volume = 0.15  # Giảm volume xuống 15% (thay vì 30%)
        self.fade_volume = 0.05    # Faded volume 5% (thay vì 10%)
        self.fade_timer = None
        self.fade_duration = 10.0  # Fade duration in seconds
        self.sound_cache = {}  # Cache để lưu pygame.mixer.Sound objects
        
    def start_background_music(self):
        """Start playing background music in loop"""
        try:
            if not self.bg_music_playing:
                pygame.mixer.music.load("asset/sound/bg-sound.mp3")
                pygame.mixer.music.set_volume(self.normal_volume)
                pygame.mixer.music.play(-1)  # Loop indefinitely
                self.bg_music_playing = True
                print(f"🎵 Background music started at {self.normal_volume*100}% volume")
        except Exception as e:
            print(f"⚠ Warning: Could not start background music: {e}")
    
    def play_countdown_sound(self):
        """Play countdown sound and fade background music"""
        try:
            # Cancel old timer if exists (người chơi nhấn SPACE liên tục)
            if self.fade_timer:
                self.fade_timer.cancel()
                print("⏱️  Timer reset - old timer cancelled")
            
            # Fade background music down
            self.fade_bg_music_down()
            
            # Play countdown sound using winsound (non-blocking)
            def _play():
                winsound.PlaySound("asset/sound/countdown.wav", 
                                 winsound.SND_FILENAME | winsound.SND_ASYNC)
            
            thread = threading.Thread(target=_play, daemon=True)
            thread.start()
            print("🔊 Countdown sound played")
            
            # Schedule fade up after 10 seconds (thay vì 5 giây)
            self.fade_timer = threading.Timer(self.fade_duration, self.fade_bg_music_up)
            self.fade_timer.start()
            print(f"⏱️  New timer started - will restore volume in {self.fade_duration}s")
            
        except Exception as e:
            print(f"⚠ Warning: Could not play countdown sound: {e}")
    
    def clear_sound_cache(self):
        """Clear sound cache để reload file mới sau khi đổi tên"""
        self.sound_cache.clear()
        print("🔄 Sound cache cleared - sẽ load file mới")
    
    def play_winner_sound(self, sound_file):
        """Play winner sound at full volume (không bị fade)"""
        try:
            def _play():
                # Load từ cache, hoặc tạo mới nếu chưa có
                if sound_file not in self.sound_cache:
                    self.sound_cache[sound_file] = pygame.mixer.Sound(sound_file)
                    print(f"📁 Loaded new sound into cache: {sound_file}")
                
                winner_sound = self.sound_cache[sound_file]
                winner_sound.set_volume(0.7)  # 70% volume - rõ ràng
                winner_sound.play()
                print(f"🎉 Winner sound played at full volume: {sound_file}")
            
            thread = threading.Thread(target=_play, daemon=True)
            thread.start()
        except Exception as e:
            print(f"⚠ Warning: Could not play winner sound: {e}")
    
    def fade_bg_music_down(self):
        """Fade background music volume down"""
        try:
            pygame.mixer.music.set_volume(self.fade_volume)
            print(f"🔉 Background music faded to {self.fade_volume*100}%")
        except Exception as e:
            print(f"⚠ Warning: Could not fade down music: {e}")
    
    def fade_bg_music_up(self):
        """Fade background music volume back up"""
        try:
            pygame.mixer.music.set_volume(self.normal_volume)
            print(f"🔊 Background music restored to {self.normal_volume*100}%")
        except Exception as e:
            print(f"⚠ Warning: Could not fade up music: {e}")
    
    def stop_all(self):
        """Stop all audio"""
        try:
            if self.fade_timer:
                self.fade_timer.cancel()
            pygame.mixer.music.stop()
            self.bg_music_playing = False
            print("🔇 All audio stopped")
        except Exception as e:
            print(f"⚠ Warning: Could not stop audio: {e}")


# =====================================
# Game Mode Selection Screen
# =====================================
class GameModeDialog(QWidget):
    """Dialog for selecting game mode"""
    mode_selected = pyqtSignal(str)  # Signal to emit game mode: "single" or "two"
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Oẳn Tù Tì - Chọn Chế Độ Chơi")
        
        # Set gradient background
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
            }
        """)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Center container
        center_container = QWidget()
        center_container.setFixedWidth(700)
        center_container.setStyleSheet("""
            QWidget {
                background: white;
                border-radius: 25px;
            }
        """)
        
        # Content layout
        content_layout = QVBoxLayout()
        content_layout.setSpacing(25)
        content_layout.setContentsMargins(50, 50, 50, 50)
        
        # FPT logo at top
        fpt_container = QHBoxLayout()
        fpt_container.setAlignment(Qt.AlignCenter)
        
        fpt_logo = QLabel()
        fpt_pixmap = QPixmap("asset/LogoFPT.png")
        if not fpt_pixmap.isNull():
            fpt_logo.setPixmap(fpt_pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        fpt_container.addWidget(fpt_logo)
        content_layout.addLayout(fpt_container)
        
        # Game icon
        icon_container = QHBoxLayout()
        icon_container.setAlignment(Qt.AlignCenter)
        
        icon_label = QLabel()
        icon_pixmap = QPixmap("asset/icons/rock-paper-scissors.png")
        if not icon_pixmap.isNull():
            icon_label.setPixmap(icon_pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        icon_container.addWidget(icon_label)
        content_layout.addLayout(icon_container)
        
        # Title
        title = QLabel("OẲN TÙ TÌ")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 48px;
            font-weight: bold;
            color: #2d3436;
            margin: 10px 0px;
        """)
        content_layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Chọn chế độ chơi")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("""
            font-size: 18px;
            color: #636e72;
            margin-bottom: 20px;
        """)
        content_layout.addWidget(subtitle)
        
        # Single Player button
        self.single_button = QPushButton("🤖 CHƠI VỚI AI")
        self.single_button.setStyleSheet("""
            QPushButton {
                padding: 25px;
                font-size: 28px;
                font-weight: bold;
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6c5ce7, stop:1 #a29bfe);
                border: none;
                border-radius: 12px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5f4dd1, stop:1 #8d84e8);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #5343c0, stop:1 #7a70d0);
            }
        """)
        self.single_button.clicked.connect(lambda: self.select_mode("single"))
        content_layout.addWidget(self.single_button)
        
        # Two Player button
        self.two_button = QPushButton("👥 HAI NGƯỜI CHƠI")
        self.two_button.setStyleSheet("""
            QPushButton {
                padding: 25px;
                font-size: 28px;
                font-weight: bold;
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00b894, stop:1 #00cec9);
                border: none;
                border-radius: 12px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00a085, stop:1 #00b5b0);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #008c75, stop:1 #009c98);
            }
        """)
        self.two_button.clicked.connect(lambda: self.select_mode("two"))
        content_layout.addWidget(self.two_button)
        
        center_container.setLayout(content_layout)
        main_layout.addWidget(center_container)
        self.setLayout(main_layout)
        
    def select_mode(self, mode):
        """Select game mode"""
        self.mode_selected.emit(mode)
        self.close()


# =====================================
# Player Name Input Screen
# =====================================
class PlayerNameDialog(QWidget):
    """Dialog for entering player names"""
    names_submitted = pyqtSignal(str, str)  # Signal to emit player names
    
    def __init__(self, mode="two"):
        super().__init__()
        self.mode = mode
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Oẳn Tù Tì - Nhập Tên Người Chơi")
        
        # Set gradient background
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
            }
        """)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Center container
        center_container = QWidget()
        center_container.setFixedWidth(700)
        center_container.setStyleSheet("""
            QWidget {
                background: white;
                border-radius: 25px;
            }
        """)
        
        # Content layout
        content_layout = QVBoxLayout()
        content_layout.setSpacing(25)
        content_layout.setContentsMargins(50, 50, 50, 50)
        
        # FPT logo at top
        fpt_container = QHBoxLayout()
        fpt_container.setAlignment(Qt.AlignCenter)
        
        fpt_logo = QLabel()
        fpt_pixmap = QPixmap("asset/LogoFPT.png")
        if not fpt_pixmap.isNull():
            fpt_logo.setPixmap(fpt_pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        fpt_container.addWidget(fpt_logo)
        content_layout.addLayout(fpt_container)
        
        # Game icon
        icon_container = QHBoxLayout()
        icon_container.setAlignment(Qt.AlignCenter)
        
        icon_label = QLabel()
        icon_pixmap = QPixmap("asset/icons/rock-paper-scissors.png")
        if not icon_pixmap.isNull():
            icon_label.setPixmap(icon_pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        icon_container.addWidget(icon_label)
        content_layout.addLayout(icon_container)
        
        # Title
        title = QLabel("OẲN TÙ TÌ")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            font-size: 48px;
            font-weight: bold;
            color: #2d3436;
            margin: 10px 0px;
        """)
        content_layout.addWidget(title)
        
        # Subtitle
        if self.mode == "single":
            subtitle_text = "Nhập tên của bạn để bắt đầu"
        else:
            subtitle_text = "Nhập tên người chơi để bắt đầu"
        subtitle = QLabel(subtitle_text)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("""
            font-size: 18px;
            color: #636e72;
            margin-bottom: 20px;
        """)
        content_layout.addWidget(subtitle)
        
        # Player 1
        if self.mode == "single":
            p1_label_text = "👤 Tên Của Bạn:"
            p1_placeholder = "Nhập tên của bạn..."
        else:
            p1_label_text = "👤 Người Chơi 1:"
            p1_placeholder = "Nhập tên Người Chơi 1..."
            
        p1_label = QLabel(p1_label_text)
        p1_label.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            color: #0984e3;
            margin-top: 10px;
        """)
        content_layout.addWidget(p1_label)
        
        self.player1_input = QLineEdit()
        self.player1_input.setPlaceholderText(p1_placeholder)
        self.player1_input.setStyleSheet("""
            QLineEdit {
                padding: 15px 20px;
                font-size: 18px;
                border: 2px solid #0984e3;
                border-radius: 10px;
                background: #f8f9fa;
                color: #2d3436;
            }
            QLineEdit:focus {
                border: 2px solid #74b9ff;
                background: white;
            }
        """)
        content_layout.addWidget(self.player1_input)
        
        # Player 2 (only for two-player mode)
        if self.mode == "two":
            p2_label = QLabel("👤 Người Chơi 2:")
            p2_label.setStyleSheet("""
                font-size: 20px;
                font-weight: bold;
                color: #fd79a8;
                margin-top: 15px;
            """)
            content_layout.addWidget(p2_label)
            
            self.player2_input = QLineEdit()
            self.player2_input.setPlaceholderText("Nhập tên Người Chơi 2...")
            self.player2_input.setStyleSheet("""
                QLineEdit {
                    padding: 15px 20px;
                    font-size: 18px;
                    border: 2px solid #fd79a8;
                    border-radius: 10px;
                    background: #f8f9fa;
                    color: #2d3436;
                }
                QLineEdit:focus {
                    border: 2px solid #fab1a0;
                    background: white;
                }
            """)
            content_layout.addWidget(self.player2_input)
        else:
            # Create dummy input for single player mode
            self.player2_input = None
        
        # Start button
        self.start_button = QPushButton("🚀 BẮT ĐẦU")
        self.start_button.setStyleSheet("""
            QPushButton {
                padding: 18px;
                font-size: 24px;
                font-weight: bold;
                color: white;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00b894, stop:1 #00cec9);
                border: none;
                border-radius: 12px;
                margin-top: 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00a085, stop:1 #00b5b0);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #008c75, stop:1 #009c98);
            }
        """)
        self.start_button.clicked.connect(self.submit_names)
        content_layout.addWidget(self.start_button)
        
        center_container.setLayout(content_layout)
        main_layout.addWidget(center_container)
        self.setLayout(main_layout)
        
        # Connect Enter key to submit
        self.player1_input.returnPressed.connect(self.submit_names)
        if self.mode == "two" and self.player2_input:
            self.player2_input.returnPressed.connect(self.submit_names)
        
    def submit_names(self):
        """Submit player names"""
        player1_name = self.player1_input.text().strip()
        
        if self.mode == "single":
            # Single player mode
            player2_name = "AI"
            if not player1_name:
                player1_name = "Player"
        else:
            # Two player mode
            player2_name = self.player2_input.text().strip()
            if not player1_name:
                player1_name = "Player 1"
            if not player2_name:
                player2_name = "Player 2"
            
        self.names_submitted.emit(player1_name, player2_name)
        self.close()


# =====================================
# Loading Screen
# =====================================
class LoadingScreen(QWidget):
    """Loading screen with animation"""
    loading_complete = pyqtSignal()
    
    def __init__(self, player1_name, player2_name):
        super().__init__()
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.dots = 0
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Oẳn Tù Tì - Đang Tải")
        
        # Set gradient background
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
            }
        """)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Center container
        center_container = QWidget()
        center_container.setFixedWidth(800)
        center_container.setStyleSheet("""
            QWidget {
                background: white;
                border-radius: 25px;
            }
        """)
        
        # Content layout
        content_layout = QVBoxLayout()
        content_layout.setSpacing(25)
        content_layout.setContentsMargins(60, 60, 60, 60)
        
        # Game icon
        icon_container = QHBoxLayout()
        icon_container.setAlignment(Qt.AlignCenter)
        
        game_icon = QLabel()
        icon_pixmap = QPixmap("asset/icons/rock-paper-scissors.png")
        if not icon_pixmap.isNull():
            game_icon.setPixmap(icon_pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        icon_container.addWidget(game_icon)
        content_layout.addLayout(icon_container)
        
        # Loading animation icon
        loading_icon = QLabel("⏳")
        loading_icon.setAlignment(Qt.AlignCenter)
        loading_icon.setStyleSheet("font-size: 80px;")
        content_layout.addWidget(loading_icon)
        
        # Loading text
        self.loading_label = QLabel("Đang khởi tạo trò chơi")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("""
            font-size: 32px;
            font-weight: bold;
            color: #2d3436;
            margin: 15px 0px;
        """)
        content_layout.addWidget(self.loading_label)
        
        # Player names section
        names_section = QWidget()
        names_section.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #667eea, stop:1 #764ba2);
            border-radius: 15px;
            padding: 20px;
        """)
        names_layout = QHBoxLayout()
        names_layout.setAlignment(Qt.AlignCenter)
        names_layout.setSpacing(15)
        
        # FPT Logo
        fpt_logo = QLabel()
        fpt_pixmap = QPixmap("asset/LogoFPT.png")
        if not fpt_pixmap.isNull():
            fpt_logo.setPixmap(fpt_pixmap.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            names_layout.addWidget(fpt_logo)
        
        # Player names
        names_label = QLabel(f"{self.player1_name} vs {self.player2_name}")
        names_label.setStyleSheet("""
            font-size: 26px;
            font-weight: bold;
            color: white;
        """)
        names_layout.addWidget(names_label)
        
        names_section.setLayout(names_layout)
        content_layout.addWidget(names_section)
        
        # Progress text
        self.progress_label = QLabel("Đang chuẩn bị camera và AI model...")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("""
            font-size: 18px;
            color: #636e72;
            margin-top: 10px;
        """)
        content_layout.addWidget(self.progress_label)
        
        center_container.setLayout(content_layout)
        main_layout.addWidget(center_container)
        self.setLayout(main_layout)
        
        # Animation timer for dots
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_dots)
        self.animation_timer.start(500)
        
        # Start audio generation in background
        self.generate_audio_files()
        
        # Completion timer (3 seconds - enough for audio generation)
        self.completion_timer = QTimer()
        self.completion_timer.timeout.connect(self.complete_loading)
        self.completion_timer.setSingleShot(True)
        self.completion_timer.start(3000)
        
    def animate_dots(self):
        """Animate loading dots"""
        self.dots = (self.dots + 1) % 4
        dots_text = "." * self.dots
        self.loading_label.setText(f"Đang khởi tạo trò chơi{dots_text}")
    
    def generate_audio_files(self):
        """Generate audio files for players using gTTS"""
        try:
            # Ensure result directory exists
            os.makedirs("asset/result", exist_ok=True)
            
            # Generate audio for Player 1 (ghi đè file cũ)
            text_p1 = f"Chúc mừng người chơi {self.player1_name} chiến thắng"
            tts_p1 = gTTS(text=text_p1, lang='vi', slow=False)
            tts_p1.save("asset/result/player-1.mp3")
            
            # Generate audio for Player 2 (ghi đè file cũ)
            text_p2 = f"Chúc mừng người chơi {self.player2_name} chiến thắng"
            tts_p2 = gTTS(text=text_p2, lang='vi', slow=False)
            tts_p2.save("asset/result/player-2.mp3")
            
            print(f"✓ Generated audio files for {self.player1_name} and {self.player2_name}")
            self.progress_label.setText("Đã tạo file âm thanh chiến thắng!")
        except Exception as e:
            print(f"⚠ Warning: Could not generate audio files: {e}")
            self.progress_label.setText("Cảnh báo: Không thể tạo file âm thanh")
        
    def complete_loading(self):
        """Complete loading and emit signal"""
        self.animation_timer.stop()
        self.loading_complete.emit()
        self.close()


# =====================================
# Video Thread for Camera Processing
# =====================================
class VideoThread(QThread):
    """Thread for processing video frames"""
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        
    def run(self):
        """Run the thread - will be controlled by main game logic"""
        # This thread is just a placeholder for now
        # The actual game loop will run in the main thread
        pass
    
    def stop(self):
        """Stop the thread"""
        self._run_flag = False
        self.wait()


# =====================================
# Main Game Window
# =====================================
class GameWindow(QMainWindow):
    """Main game window with video feed"""
    
    def __init__(self, player1_name, player2_name, game_mode="two"):
        super().__init__()
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.game_mode = game_mode
        self.player1_score = 0
        self.player2_score = 0
        self.draws = 0
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Rock Paper Scissors - Game")
        self.setGeometry(100, 100, 1280, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Top bar with logo and scores
        top_bar = self.create_top_bar()
        main_layout.addWidget(top_bar)
        
        # Video display area
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(1280, 720)
        main_layout.addWidget(self.video_label)
        
        # Bottom control bar
        bottom_bar = self.create_bottom_bar()
        main_layout.addWidget(bottom_bar)
        
        central_widget.setLayout(main_layout)
        
    def create_top_bar(self):
        """Create top bar with logo and scores"""
        top_frame = QFrame()
        top_frame.setStyleSheet("""
            QFrame {
                background-color: #1E1E2E;
                border-bottom: 3px solid #FFD700;
            }
        """)
        top_frame.setFixedHeight(80)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 10, 20, 10)
        
        # Player 1 info
        self.player1_label = QLabel(f"👤 {self.player1_name}")
        self.player1_label.setStyleSheet("""
            QLabel {
                color: #00FFFF;
                font-size: 20px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.player1_label)
        
        self.player1_score_label = QLabel("0")
        self.player1_score_label.setStyleSheet("""
            QLabel {
                color: #00FFFF;
                font-size: 28px;
                font-weight: bold;
                padding: 0 10px;
            }
        """)
        layout.addWidget(self.player1_score_label)
        
        # Center - Logo and draws
        layout.addStretch()
        
        center_layout = QVBoxLayout()
        center_layout.setSpacing(2)
        
        logo_label = QLabel("🎮 FPT RPS 🎮")
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setStyleSheet("""
            QLabel {
                color: #FFD700;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        center_layout.addWidget(logo_label)
        
        self.draws_label = QLabel("Hòa: 0")
        self.draws_label.setAlignment(Qt.AlignCenter)
        self.draws_label.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-size: 14px;
            }
        """)
        center_layout.addWidget(self.draws_label)
        
        layout.addLayout(center_layout)
        layout.addStretch()
        
        # Player 2 info
        self.player2_score_label = QLabel("0")
        self.player2_score_label.setStyleSheet("""
            QLabel {
                color: #FFA500;
                font-size: 28px;
                font-weight: bold;
                padding: 0 10px;
            }
        """)
        layout.addWidget(self.player2_score_label)
        
        self.player2_label = QLabel(f"{self.player2_name} 👤")
        self.player2_label.setStyleSheet("""
            QLabel {
                color: #FFA500;
                font-size: 20px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.player2_label)
        
        top_frame.setLayout(layout)
        return top_frame
    
    def create_bottom_bar(self):
        """Create bottom control bar"""
        bottom_frame = QFrame()
        bottom_frame.setStyleSheet("""
            QFrame {
                background-color: #1E1E2E;
                border-top: 3px solid #FFD700;
            }
        """)
        bottom_frame.setFixedHeight(60)
        
        layout = QHBoxLayout()
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(20)
        
        # Instructions
        self.instructions = QLabel("⌨️ SPACE: Bắt đầu  |  R: Reset điểm  |  ESC: Menu  |  Q: Thoát")
        self.instructions.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-size: 14px;
            }
        """)
        layout.addWidget(self.instructions)
        
        layout.addStretch()
        
        # Status label
        self.status_label = QLabel("Sẵn sàng chơi!")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #00FF00;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.status_label)
        
        bottom_frame.setLayout(layout)
        return bottom_frame
    
    def update_frame(self, frame):
        """Update video frame display"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            
            # Convert to QImage
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale to fit label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            self.video_label.setPixmap(scaled_pixmap)
        except Exception as e:
            print(f"Error updating frame: {e}")
    
    def update_scores(self, player1_score, player2_score, draws):
        """Update score displays"""
        self.player1_score = player1_score
        self.player2_score = player2_score
        self.draws = draws
        
        self.player1_score_label.setText(str(player1_score))
        self.player2_score_label.setText(str(player2_score))
        self.draws_label.setText(f"Hòa: {draws}")
        
        # Update player labels with winning indicator
        if player1_score > player2_score:
            self.player1_label.setText(f"👤 {self.player1_name} 🏆")
        else:
            self.player1_label.setText(f"👤 {self.player1_name}")
            
        if player2_score > player1_score:
            self.player2_label.setText(f"🏆 {self.player2_name} 👤")
        else:
            self.player2_label.setText(f"{self.player2_name} 👤")
    
    def update_status(self, status_text, color="#00FF00"):
        """Update status label"""
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 16px;
                font-weight: bold;
            }}
        """)
    
    def keyPressEvent(self, event):
        """Handle key press events"""
        # This will be handled by the main game logic
        super().keyPressEvent(event)


# =====================================
# Application Manager
# =====================================
class RPSApplication:
    """Manages the entire application flow"""
    
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.player1_name = "Player 1"
        self.player2_name = "Player 2"
        self.game_mode = "two"  # "single" or "two"
        self.game_window = None
        self.audio_manager = AudioManager()
        
    def start(self):
        """Start the application"""
        # Start background music
        self.audio_manager.start_background_music()
        
        # Show game mode selection dialog
        self.show_mode_dialog()
    
    def show_mode_dialog(self):
        """Show game mode selection dialog"""
        self.mode_dialog = GameModeDialog()
        # Check if we're returning from game (reconnect signals)
        if hasattr(self, 'game_instance'):
            self.mode_dialog.mode_selected.connect(self.on_menu_return_mode_selected)
        else:
            self.mode_dialog.mode_selected.connect(self.on_mode_selected)
        self.mode_dialog.showFullScreen()
    
    def on_mode_selected(self, mode):
        """Handle game mode selection"""
        self.game_mode = mode
        # Show name input dialog
        self.show_name_dialog()
        
    def show_name_dialog(self):
        """Show player name input dialog"""
        self.name_dialog = PlayerNameDialog(mode=self.game_mode)
        # Check if we're returning from game (reconnect signals)
        if hasattr(self, 'game_instance'):
            self.name_dialog.names_submitted.connect(self.on_menu_return_names_submitted)
        else:
            self.name_dialog.names_submitted.connect(self.on_names_submitted)
        self.name_dialog.showFullScreen()
    
    def show_name_dialog_for_restart(self, game_instance):
        """Show player name input dialog for restart"""
        self.game_instance = game_instance
        self.name_dialog = PlayerNameDialog(mode=self.game_mode)
        self.name_dialog.names_submitted.connect(self.on_names_submitted_restart)
        self.name_dialog.showFullScreen()
        
    def on_names_submitted(self, player1_name, player2_name):
        """Handle player names submission"""
        self.player1_name = player1_name
        self.player2_name = player2_name
        
        # Show loading screen
        self.show_loading_screen()
    
    def on_names_submitted_restart(self, player1_name, player2_name):
        """Handle player names submission for restart"""
        self.player1_name = player1_name
        self.player2_name = player2_name
        
        # Show loading screen for restart
        self.loading_screen = LoadingScreen(self.player1_name, self.player2_name)
        self.loading_screen.loading_complete.connect(self.on_restart_loading_complete)
        self.loading_screen.showFullScreen()
    
    def on_restart_loading_complete(self):
        """Handle restart loading completion"""
        # Clear sound cache để load file âm thanh mới
        self.audio_manager.clear_sound_cache()
        
        # Update player names in game instance
        if hasattr(self, 'game_instance'):
            self.game_instance.player1.name = self.player1_name
            if self.game_mode == "two" and self.game_instance.player2:
                self.game_instance.player2.name = self.player2_name
            
            # Update game window with new names
            self.game_instance.game_window.player1_name = self.player1_name
            self.game_instance.game_window.player2_name = self.player2_name
            self.game_instance.game_window.player1_label.setText(f"👤 {self.player1_name}")
            self.game_instance.game_window.player2_label.setText(f"{self.player2_name} 👤")
            
            # Show game window again
            self.game_instance.game_window.show()
            
            # Restart timer
            self.game_instance.timer.start(self.game_instance.timer_interval)
            
            self.game_instance.game_window.update_status("Đã cập nhật tên người chơi!", "#00FF00")
    
    def on_menu_return_mode_selected(self, mode):
        """Handle mode selection when returning from menu"""
        self.game_mode = mode
        # Show name input dialog
        self.show_name_dialog()
        
    def on_menu_return_names_submitted(self, player1_name, player2_name):
        """Handle names submission when returning from menu"""
        self.player1_name = player1_name
        self.player2_name = player2_name
        
        # Show loading screen
        self.loading_screen = LoadingScreen(self.player1_name, self.player2_name)
        self.loading_screen.loading_complete.connect(self.on_menu_return_loading_complete)
        self.loading_screen.showFullScreen()
    
    def on_menu_return_loading_complete(self):
        """Handle loading completion when returning from menu"""
        # Reinitialize the game with new settings
        if hasattr(self, 'on_loading_complete') and callable(self.on_loading_complete):
            self.on_loading_complete()
        
    def show_loading_screen(self):
        """Show loading screen"""
        self.loading_screen = LoadingScreen(self.player1_name, self.player2_name)
        self.loading_screen.loading_complete.connect(self.on_loading_complete)
        self.loading_screen.showFullScreen()
        
    def on_loading_complete(self):
        """Handle loading completion"""
        # This will be connected to start the actual game
        pass
    
    def show_game_window(self):
        """Show main game window"""
        self.game_window = GameWindow(self.player1_name, self.player2_name, self.game_mode)
        self.game_window.show()
        return self.game_window
    
    def exec(self):
        """Execute the application"""
        return self.app.exec_()
