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


# =====================================
# Player Name Input Screen
# =====================================
class PlayerNameDialog(QWidget):
    """Dialog for entering player names"""
    names_submitted = pyqtSignal(str, str)  # Signal to emit player names
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """Initialize the UI"""
        self.setWindowTitle("Rock Paper Scissors - Player Names")
        self.setFixedSize(600, 450)
        
        # Set background color
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(30, 30, 40))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(50, 40, 50, 40)
        
        # Title
        title = QLabel("🎮 ROCK PAPER SCISSORS 🎮")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                color: #FFD700;
                font-size: 28px;
                font-weight: bold;
                padding: 20px;
            }
        """)
        main_layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Nhập tên người chơi để bắt đầu")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-size: 16px;
                padding: 10px;
            }
        """)
        main_layout.addWidget(subtitle)
        
        # Player 1 input
        player1_label = QLabel("👤 Player 1:")
        player1_label.setStyleSheet("""
            QLabel {
                color: #00FFFF;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        main_layout.addWidget(player1_label)
        
        self.player1_input = QLineEdit()
        self.player1_input.setPlaceholderText("Nhập tên Player 1...")
        self.player1_input.setStyleSheet("""
            QLineEdit {
                background-color: #1E1E2E;
                color: #FFFFFF;
                border: 2px solid #00FFFF;
                border-radius: 8px;
                padding: 10px 15px;
                font-size: 15px;
                min-height: 20px;
            }
            QLineEdit:focus {
                border: 2px solid #FFD700;
            }
        """)
        main_layout.addWidget(self.player1_input)
        
        # Player 2 input
        player2_label = QLabel("👤 Player 2:")
        player2_label.setStyleSheet("""
            QLabel {
                color: #FFA500;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        main_layout.addWidget(player2_label)
        
        self.player2_input = QLineEdit()
        self.player2_input.setPlaceholderText("Nhập tên Player 2...")
        self.player2_input.setStyleSheet("""
            QLineEdit {
                background-color: #1E1E2E;
                color: #FFFFFF;
                border: 2px solid #FFA500;
                border-radius: 8px;
                padding: 10px 15px;
                font-size: 15px;
                min-height: 20px;
            }
            QLineEdit:focus {
                border: 2px solid #FFD700;
            }
        """)
        main_layout.addWidget(self.player2_input)
        
        # Start button
        self.start_button = QPushButton("🚀 BẮT ĐẦU")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #28A745;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 15px;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1E7E34;
            }
        """)
        self.start_button.clicked.connect(self.submit_names)
        main_layout.addWidget(self.start_button)
        
        # Add stretch to push everything up
        main_layout.addStretch()
        
        self.setLayout(main_layout)
        
        # Connect Enter key to submit
        self.player1_input.returnPressed.connect(self.submit_names)
        self.player2_input.returnPressed.connect(self.submit_names)
        
    def submit_names(self):
        """Submit player names"""
        player1_name = self.player1_input.text().strip()
        player2_name = self.player2_input.text().strip()
        
        # Use default names if empty
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
        self.setWindowTitle("Rock Paper Scissors - Loading")
        self.setFixedSize(800, 500)
        
        # Set background color
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(20, 20, 30))
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(30)
        main_layout.setContentsMargins(50, 80, 50, 80)
        
        # Loading icon
        icon_label = QLabel("⏳")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("""
            QLabel {
                font-size: 80px;
            }
        """)
        main_layout.addWidget(icon_label)
        
        # Loading text
        self.loading_label = QLabel("Đang khởi tạo trò chơi")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("""
            QLabel {
                color: #00FFFF;
                font-size: 28px;
                font-weight: bold;
                padding: 20px;
            }
        """)
        main_layout.addWidget(self.loading_label)
        
        # Player names
        names_label = QLabel(f"🎮 {self.player1_name} vs {self.player2_name} 🎮")
        names_label.setAlignment(Qt.AlignCenter)
        names_label.setStyleSheet("""
            QLabel {
                color: #FFD700;
                font-size: 20px;
                font-weight: bold;
                padding: 10px;
            }
        """)
        main_layout.addWidget(names_label)
        
        # Progress text
        self.progress_label = QLabel("Đang chuẩn bị camera và AI model...")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("""
            QLabel {
                color: #AAAAAA;
                font-size: 14px;
                padding: 10px;
            }
        """)
        main_layout.addWidget(self.progress_label)
        
        # Add stretch
        main_layout.addStretch()
        
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
            
            # Generate audio for Player 1
            text_p1 = f"Chúc mừng người chơi {self.player1_name} chiến thắng"
            tts_p1 = gTTS(text=text_p1, lang='vi', slow=False)
            tts_p1.save("asset/result/player-1.mp3")
            
            # Generate audio for Player 2
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
    
    def __init__(self, player1_name, player2_name):
        super().__init__()
        self.player1_name = player1_name
        self.player2_name = player2_name
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
        instructions = QLabel("⌨️ SPACE: Bắt đầu  |  R: Reset điểm  |  N: Đổi tên  |  Q: Thoát")
        instructions.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-size: 14px;
            }
        """)
        layout.addWidget(instructions)
        
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
        self.game_window = None
        
    def start(self):
        """Start the application"""
        # Show name input dialog
        self.show_name_dialog()
        
    def show_name_dialog(self):
        """Show player name input dialog"""
        self.name_dialog = PlayerNameDialog()
        self.name_dialog.names_submitted.connect(self.on_names_submitted)
        self.name_dialog.show()
    
    def show_name_dialog_for_restart(self, game_instance):
        """Show player name input dialog for restart"""
        self.game_instance = game_instance
        self.name_dialog = PlayerNameDialog()
        self.name_dialog.names_submitted.connect(self.on_names_submitted_restart)
        self.name_dialog.show()
        
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
        self.loading_screen.show()
    
    def on_restart_loading_complete(self):
        """Handle restart loading completion"""
        # Update player names in game instance
        if hasattr(self, 'game_instance'):
            self.game_instance.player1.name = self.player1_name
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
            
            self.game_instance.game_window.update_status("Trò chơi đã khởi động lại!", "#00FF00")
        
    def show_loading_screen(self):
        """Show loading screen"""
        self.loading_screen = LoadingScreen(self.player1_name, self.player2_name)
        self.loading_screen.loading_complete.connect(self.on_loading_complete)
        self.loading_screen.show()
        
    def on_loading_complete(self):
        """Handle loading completion"""
        # This will be connected to start the actual game
        pass
    
    def show_game_window(self):
        """Show main game window"""
        self.game_window = GameWindow(self.player1_name, self.player2_name)
        self.game_window.show()
        return self.game_window
    
    def exec(self):
        """Execute the application"""
        return self.app.exec_()
