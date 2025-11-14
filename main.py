# =====================================
# Rock Paper Scissors - 2 Player Game (Multi-threaded)
# =====================================
import cv2
import numpy as np
import mediapipe as mp
from joblib import load
import time
import threading
from collections import deque
import winsound  # Windows native audio - ultra fast!

# Import module chung
from hand_feature_extractor import HandFeatureExtractor


# =====================================
# Audio Helper
# =====================================
def play_sound(wav_file):
    """Play WAV file using Windows native API - ultra fast and non-blocking"""
    def _play():
        try:
            # SND_ASYNC = play and return immediately (fire-and-forget)
            # SND_FILENAME = interpret wav_file as a filename
            winsound.PlaySound(wav_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as e:
            print(f"⚠ Warning: Could not play audio {wav_file}: {e}")

    # Still use thread to be extra safe, but winsound is already non-blocking
    thread = threading.Thread(target=_play, daemon=True)
    thread.start()


# =====================================
# Player class with separate Mediapipe instance
# =====================================
class Player:
    """Each player has their own Mediapipe hands instance and processing thread"""

    def __init__(self, player_id, name, model, scaler):
        self.player_id = player_id
        self.name = name
        self.model = model
        self.scaler = scaler

        # Separate HandFeatureExtractor instance for this player
        self.feature_extractor = HandFeatureExtractor(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,  # Lower for faster detection
            min_tracking_confidence=0.5    # Lower for smoother tracking
        )

        # For drawing landmarks
        self.mp_hands = self.feature_extractor.mp_hands
        self.mp_drawing = self.feature_extractor.mp_drawing

        # Thread-safe data storage
        self.lock = threading.Lock()
        self.frame = None
        self.features = None
        self.landmarks = None  # For drawing
        self.prediction = None
        self.prediction_buffer = deque(maxlen=5)  # Smoothing buffer (reduced from 7)
        self.captured_frame = None  # Store captured frame from previous round
        self.captured_gesture = None  # Store captured gesture from previous round
        self.captured_frame_with_landmarks = None  # Store frame WITH MediaPipe drawing

        # Thread control
        self.running = False
        self.thread = None
        self.frame_ready = threading.Event()  # Event-based processing
        self.game_mode = "play"  # Track game mode for adaptive processing

    def start(self):
        """Start the processing thread"""
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the processing thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.feature_extractor.close()

    def update_frame(self, frame, game_mode="play"):
        """Update the frame to be processed (thread-safe)"""
        with self.lock:
            self.frame = frame  # No copy needed here, thread will handle it
            self.game_mode = game_mode  # Update game mode
        self.frame_ready.set()  # Signal that new frame is ready

    def get_results(self):
        """Get the latest processing results (thread-safe)"""
        with self.lock:
            return {
                'features': self.features,
                'landmarks': self.landmarks,
                'prediction': self.prediction
            }

    def _process_loop(self):
        """Main processing loop running in separate thread - OPTIMIZED"""
        frame_skip_counter = 0
        
        while self.running:
            # Wait for new frame instead of polling
            if not self.frame_ready.wait(timeout=0.1):
                continue
            self.frame_ready.clear()

            # Get frame to process
            with self.lock:
                if self.frame is None:
                    continue
                frame_to_process = self.frame
                current_game_mode = self.game_mode
            
            # Adaptive frame skipping based on game mode
            frame_skip_counter += 1
            if current_game_mode == "countdown":
                # During countdown, process every 3rd frame to save CPU
                if frame_skip_counter % 3 != 0:
                    continue
            elif current_game_mode == "result":
                # During result display, process every 5th frame
                if frame_skip_counter % 5 != 0:
                    continue
            # In "play" mode, process every frame for real-time feedback

            # Process frame with SINGLE MediaPipe call
            features, landmarks = self._extract_features_optimized(frame_to_process)

            # Predict gesture
            prediction = None
            if features is not None:
                prediction = self._predict_gesture(features)
                self.prediction_buffer.append(prediction)

            # Get smoothed prediction
            smoothed_prediction = self._get_smoothed_prediction()

            # Store results
            with self.lock:
                self.features = features
                self.landmarks = landmarks
                self.prediction = smoothed_prediction

    def _extract_features_optimized(self, frame):
        """Extract features with SINGLE MediaPipe call - OPTIMIZED"""
        # Convert to RGB once
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # SINGLE MediaPipe call - this is the bottleneck!
        results = self.feature_extractor.hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None, None

        hand_landmarks = results.multi_hand_landmarks[0]

        # Extract landmarks as numpy array
        landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

        # Normalize hand orientation (IMPORTANT for model accuracy)
        normalized = normalize_hand_orientation(landmarks_array)

        # Extract features from normalized landmarks
        features = self.feature_extractor.extract_features_from_landmarks(normalized)

        return features, hand_landmarks  # Return both features and landmarks for drawing

    def _predict_gesture(self, features):
        """Predict gesture from features"""
        if features is None:
            return None

        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]

        labels = {0: "Rock", 1: "Paper", 2: "Scissors"}
        return labels[prediction]

    def _get_smoothed_prediction(self):
        """Get most common prediction from buffer - OPTIMIZED"""
        if not self.prediction_buffer:
            return None

        # Use Counter for faster counting
        from collections import Counter
        counts = Counter(pred for pred in self.prediction_buffer if pred)
        
        if not counts:
            return None

        # Return most common
        return counts.most_common(1)[0][0]


# =====================================
# Helper functions
# =====================================
def normalize_hand_orientation(landmarks):
    """
    Normalize hand orientation to match training data (hand pointing up)
    OPTIMIZED: Use vectorized operations for 5-10x speedup
    """
    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    hand_vector = middle_mcp - wrist
    hand_vector_2d = hand_vector[:2]

    current_angle = np.arctan2(hand_vector_2d[0], -hand_vector_2d[1])

    cos_angle = np.cos(-current_angle)
    sin_angle = np.sin(-current_angle)
    rotation_matrix_2d = np.array([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])

    # OPTIMIZED: Vectorized rotation for all points at once
    rotated_landmarks = landmarks.copy()
    points_2d = landmarks[:, :2] - wrist[:2]  # Center all points at once
    rotated_points_2d = points_2d @ rotation_matrix_2d.T  # Vectorized rotation
    rotated_landmarks[:, :2] = rotated_points_2d + wrist[:2]  # Translate back
    # Keep z-coordinate unchanged
    rotated_landmarks[:, 2] = landmarks[:, 2]

    return rotated_landmarks


def draw_captured_frame(frame, captured_frame, position="top-left", gesture_text=""):
    """Draw captured frame from previous round with gesture label"""
    # Only draw if there is a captured frame
    if captured_frame is None:
        return frame

    viz_size = 150  # Reduced from 200 to 150
    h, w = frame.shape[:2]
    margin = 10

    if position == "top-left":
        x_offset, y_offset = margin, margin
    else:
        x_offset, y_offset = w - viz_size - margin, margin

    # Resize captured frame to fit
    viz_img = cv2.resize(captured_frame, (viz_size, viz_size))

    # Add semi-transparent background for text
    overlay = viz_img.copy()
    cv2.rectangle(overlay, (0, viz_size - 35), (viz_size, viz_size), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, viz_img, 0.3, 0, viz_img)

    # Add gesture text
    if gesture_text:
        cv2.putText(viz_img, "Last Round:", (5, viz_size - 22),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(viz_img, gesture_text, (5, viz_size - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw border
    cv2.rectangle(frame, (x_offset-2, y_offset-2),
                  (x_offset+viz_size+2, y_offset+viz_size+2),
                  (255, 255, 255), 2)

    # Place on frame
    frame[y_offset:y_offset+viz_size, x_offset:x_offset+viz_size] = viz_img

    return frame


def draw_logo(frame, logo_img, position="top-center", max_height=80):
    """Draw FPT logo on frame"""
    if logo_img is None:
        return frame

    h, w = frame.shape[:2]

    # Resize logo maintaining aspect ratio
    logo_h, logo_w = logo_img.shape[:2]
    aspect_ratio = logo_w / logo_h
    new_height = max_height
    new_width = int(new_height * aspect_ratio)

    logo_resized = cv2.resize(logo_img, (new_width, new_height))

    # Position logo
    if position == "top-center":
        x_offset = (w - new_width) // 2
        y_offset = 10
    elif position == "top-right":
        x_offset = w - new_width - 10
        y_offset = 10
    else:  # top-left
        x_offset = 10
        y_offset = 10

    # Handle transparency if logo has alpha channel
    if logo_resized.shape[2] == 4:
        # Extract alpha channel
        alpha = logo_resized[:, :, 3] / 255.0

        # Get ROI from frame
        roi = frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width]

        # Blend logo with frame
        for c in range(3):
            roi[:, :, c] = (alpha * logo_resized[:, :, c] +
                           (1 - alpha) * roi[:, :, c])

        frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = roi
    else:
        # No alpha channel, just overlay
        frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = logo_resized

    return frame


def determine_winner(player1_gesture, player2_gesture):
    """Determine the winner of the game"""
    if player1_gesture is None or player2_gesture is None:
        return None

    if player1_gesture == player2_gesture:
        return "draw"

    win_conditions = {
        ("Rock", "Scissors"): "p1",
        ("Scissors", "Paper"): "p1",
        ("Paper", "Rock"): "p1",
        ("Scissors", "Rock"): "p2",
        ("Paper", "Scissors"): "p2",
        ("Rock", "Paper"): "p2"
    }

    return win_conditions.get((player1_gesture, player2_gesture), None)


# =====================================
# RPS Game Class
# =====================================
class RPSGame:
    """Main game class for Rock Paper Scissors"""

    def __init__(self, model_path, scaler_path, camera_width=1280, camera_height=720, countdown_duration=3):
        """
        Initialize game with model and scaler

        Args:
            model_path: Path to trained model
            scaler_path: Path to feature scaler
            camera_width: Width of camera capture
            camera_height: Height of camera capture
            countdown_duration: Duration of countdown in seconds
        """
        # Load trained model and scaler
        self.model = load(model_path)
        self.scaler = load(scaler_path)

        # Game configuration
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.countdown_duration = countdown_duration

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

    def run(self):
        """Run the main game loop - OPTIMIZED"""
        # Initialize camera with optimized settings for Windows
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow = faster, more stable on Windows
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Limit to 30 FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer = less lag
        
        # Disable auto features for stable, consistent FPS (reduces jitter)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure mode
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Disable auto white balance
        
        # Frame timing for consistent FPS (prevents jitter)
        target_fps = 30
        frame_time = 1.0 / target_fps
        last_frame_time = time.time()
        
        print(f"📷 Camera initialized: {self.camera_width}x{self.camera_height} @ {target_fps}FPS")

        # Create two players with separate threads
        player1 = Player(1, "Player 1", self.model, self.scaler)
        player2 = Player(2, "Player 2", self.model, self.scaler)

        # Start processing threads
        player1.start()
        player2.start()

        # Game state
        game_mode = "play"
        countdown_start = None
        player1_final = None
        player2_final = None
        result = ""
        result_time = None

        # Score tracking
        player1_score = 0
        player2_score = 0
        draws = 0

        print("🎮 Rock Paper Scissors Game")
        print("Controls: SPACE=Start | R=Reset | Q=Quit")

        try:
            while True:
                # Frame timing control: maintain consistent 30 FPS to reduce jitter
                current_time = time.time()
                elapsed_time = current_time - last_frame_time
                
                # Skip if we're running too fast (prevents unnecessary CPU usage)
                if elapsed_time < frame_time:
                    time.sleep(frame_time - elapsed_time)
                    continue
                    
                last_frame_time = current_time
                
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Failed to read frame from camera")
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                mid_width = width // 2

                # Split frame into two halves (use slicing, not copy)
                frame_left = frame[:, :mid_width]
                frame_right = frame[:, mid_width:]
                
                # Store CLEAN frames for potential capture (before drawing landmarks)
                clean_frame_left = frame_left.copy()
                clean_frame_right = frame_right.copy()

                # Update frames for each player (threads will process them)
                player1.update_frame(frame_left, game_mode)
                player2.update_frame(frame_right, game_mode)

                # Get results from both players
                results_p1 = player1.get_results()
                results_p2 = player2.get_results()

                # Draw hand landmarks
                if results_p1['landmarks']:
                    player1.mp_drawing.draw_landmarks(
                        frame_left, results_p1['landmarks'], player1.mp_hands.HAND_CONNECTIONS,
                        player1.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        player1.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )

                # Draw captured frame from previous round (only if exists)
                # Use frame WITH landmarks for accurate display
                if player1.captured_frame_with_landmarks is not None:
                    frame_left = draw_captured_frame(frame_left, player1.captured_frame_with_landmarks, "top-left", player1.captured_gesture)

                if results_p2['landmarks']:
                    player2.mp_drawing.draw_landmarks(
                        frame_right, results_p2['landmarks'], player2.mp_hands.HAND_CONNECTIONS,
                        player2.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        player2.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )

                # Draw captured frame from previous round (only if exists)
                # Use frame WITH landmarks for accurate display
                if player2.captured_frame_with_landmarks is not None:
                    frame_right = draw_captured_frame(frame_right, player2.captured_frame_with_landmarks, "top-right", player2.captured_gesture)

                # Game logic
                if game_mode == "play":
                    gesture_p1 = results_p1['prediction'] if results_p1['prediction'] else "No hand"
                    gesture_p2 = results_p2['prediction'] if results_p2['prediction'] else "No hand"

                    cv2.putText(frame_left, f"Player 1", (10, 250),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame_left, f"{gesture_p1}", (10, 290),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.putText(frame_right, f"Player 2", (10, 250),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame_right, f"{gesture_p2}", (10, 290),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                elif game_mode == "countdown":
                    elapsed = time.time() - countdown_start
                    remaining = self.countdown_duration - elapsed

                    if remaining > 0:
                        countdown_text = str(int(remaining) + 1)

                        cv2.putText(frame_left, countdown_text, (mid_width//2 - 50, height//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
                        cv2.putText(frame_right, countdown_text, (mid_width//2 - 50, height//2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
                    else:
                        # CAPTURE IMMEDIATELY at T=3.0s and FORCE threads to process!
                        if not hasattr(self, '_capture_done'):
                            # Capture CLEAN frames (no MediaPipe drawings from live processing)
                            self._captured_frame_left = clean_frame_left.copy()
                            self._captured_frame_right = clean_frame_right.copy()
                            
                            # FORCE threads to process these exact frames from scratch
                            # Clear old predictions and buffer (thread-safe with lock)
                            with player1.lock:
                                player1.prediction_buffer.clear()
                            with player2.lock:
                                player2.prediction_buffer.clear()
                            
                            # Send captured frames to threads for processing
                            player1.update_frame(self._captured_frame_left, game_mode)
                            player2.update_frame(self._captured_frame_right, game_mode)
                            
                            self._capture_time = time.time()
                            self._capture_done = True
                            self._predictions_ready = False
                            print("📸 Captured at T=3.0s - Sending to threads for processing...")
                        
                        # Wait for threads to process the captured frames
                        processing_time = time.time() - self._capture_time
                        
                        if processing_time < 0.3:
                            # Show "PROCESSING..." while waiting
                            cv2.putText(frame_left, "PROCESSING...", (mid_width//2 - 200, height//2),
                                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 6)
                            cv2.putText(frame_right, "PROCESSING...", (mid_width//2 - 200, height//2),
                                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 6)
                            # Continue sending the same captured frames to ensure processing
                            player1.update_frame(self._captured_frame_left, game_mode)
                            player2.update_frame(self._captured_frame_right, game_mode)
                        else:
                            # After 0.3s, check if we have new predictions from captured frames
                            if not self._predictions_ready:
                                # Get fresh results after processing captured frames
                                results_p1 = player1.get_results()
                                results_p2 = player2.get_results()
                                
                                pred_p1 = results_p1['prediction']
                                pred_p2 = results_p2['prediction']
                                landmarks_p1 = results_p1['landmarks']
                                landmarks_p2 = results_p2['landmarks']
                                
                                # Check if both have valid predictions from captured frames
                                if (pred_p1 is None or pred_p2 is None or 
                                    landmarks_p1 is None or landmarks_p2 is None):
                                    if processing_time < 2.0:  # Max wait 2s
                                        # Show warning and keep processing
                                        if pred_p1 is None or landmarks_p1 is None:
                                            cv2.putText(frame_left, "DETECTING...", (50, height//2),
                                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)
                                        if pred_p2 is None or landmarks_p2 is None:
                                            cv2.putText(frame_right, "DETECTING...", (50, height//2),
                                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)
                                        # Keep sending captured frames
                                        player1.update_frame(self._captured_frame_left, game_mode)
                                        player2.update_frame(self._captured_frame_right, game_mode)
                                        continue
                                    else:
                                        # Timeout - use whatever we have
                                        pred_p1 = pred_p1 if pred_p1 else None
                                        pred_p2 = pred_p2 if pred_p2 else None
                                
                                # Store results from captured frames
                                # Create frames WITH MediaPipe landmarks for display
                                captured_with_landmarks_left = self._captured_frame_left.copy()
                                captured_with_landmarks_right = self._captured_frame_right.copy()
                                
                                # Draw landmarks on captured frames
                                if landmarks_p1:
                                    player1.mp_drawing.draw_landmarks(
                                        captured_with_landmarks_left, landmarks_p1, player1.mp_hands.HAND_CONNECTIONS,
                                        player1.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                        player1.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                                    )
                                if landmarks_p2:
                                    player2.mp_drawing.draw_landmarks(
                                        captured_with_landmarks_right, landmarks_p2, player2.mp_hands.HAND_CONNECTIONS,
                                        player2.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                        player2.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                                    )
                                
                                # Store both raw and processed frames
                                player1.captured_frame = self._captured_frame_left
                                player2.captured_frame = self._captured_frame_right
                                player1.captured_frame_with_landmarks = captured_with_landmarks_left
                                player2.captured_frame_with_landmarks = captured_with_landmarks_right
                                player1.captured_gesture = pred_p1 if pred_p1 else "No hand"
                                player2.captured_gesture = pred_p2 if pred_p2 else "No hand"

                                player1_final = pred_p1
                                player2_final = pred_p2
                                winner = determine_winner(player1_final, player2_final)

                                if winner == "p1":
                                    player1_score += 1
                                    result = "Player 1 Wins!"
                                    play_sound("asset/result/player-1.wav")
                                elif winner == "p2":
                                    player2_score += 1
                                    result = "Player 2 Wins!"
                                    play_sound("asset/result/player-2.wav")
                                elif winner == "draw":
                                    draws += 1
                                    result = "Draw!"
                                    play_sound("asset/result/tie.wav")
                                else:
                                    result = "No hands detected!"

                                result_time = time.time()
                                game_mode = "result"
                                # Reset flags for next round
                                delattr(self, '_capture_done')
                                delattr(self, '_predictions_ready')

                elif game_mode == "result":
                    cv2.putText(frame_left, f"Player 1", (10, 250),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame_left, f"{player1_final if player1_final else 'No hand'}",
                               (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.putText(frame_right, f"Player 2", (10, 250),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame_right, f"{player2_final if player2_final else 'No hand'}",
                               (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if time.time() - result_time > 3:
                        game_mode = "play"
                        player1_final = None
                        player2_final = None
                        result = ""

                # Combine frames
                combined_frame = np.hstack([frame_left, frame_right])

                # Draw center line
                cv2.line(combined_frame, (mid_width, 0), (mid_width, height), (255, 255, 255), 2)

                # Draw FPT logo at the top center
                combined_frame = draw_logo(combined_frame, self.logo, position="top-center", max_height=60)

                # Display scores
                score_text = f"P1: {player1_score}  |  Draws: {draws}  |  P2: {player2_score}"
                text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                score_x = (width - text_size[0]) // 2
                score_y = 80  # Moved down to avoid logo

                cv2.rectangle(combined_frame,
                             (score_x - 10, score_y - text_size[1] - 10),
                             (score_x + text_size[0] + 10, score_y + 10),
                             (0, 0, 0), -1)

                cv2.putText(combined_frame, score_text, (score_x, score_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

                # Highlight winner
                if player1_score > player2_score:
                    cv2.putText(combined_frame, "WINNING!", (50, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                elif player2_score > player1_score:
                    cv2.putText(combined_frame, "WINNING!", (width - 250, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Display result
                if result:
                    text_size = cv2.getTextSize(result, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
                    text_x = (width - text_size[0]) // 2
                    text_y = height - 50

                    cv2.rectangle(combined_frame,
                                 (text_x - 10, text_y - text_size[1] - 10),
                                 (text_x + text_size[0] + 10, text_y + 10),
                                 (0, 0, 0), -1)

                    if "Player 1" in result:
                        color = (0, 255, 255)
                    elif "Player 2" in result:
                        color = (0, 165, 255)
                    else:
                        color = (0, 255, 0)

                    cv2.putText(combined_frame, result, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

                # Display instructions
                cv2.putText(combined_frame, "SPACE: Start | R: Reset | Q: Quit | OPTIMIZED Mode",
                           (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow("Rock Paper Scissors - 2 Players (OPTIMIZED)", combined_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord(' ') and game_mode == "play":
                    game_mode = "countdown"
                    countdown_start = time.time()
                    result = ""
                elif key == ord('r'):
                    game_mode = "play"
                    player1_final = None
                    player2_final = None
                    result = ""
                    player1_score = 0
                    player2_score = 0
                    draws = 0

        finally:
            # Cleanup
            player1.stop()
            player2.stop()
            cap.release()
            cv2.destroyAllWindows()

            # Print final scores
            print("\n🏆 FINAL SCORES")
            print(f"Player 1: {player1_score} | Player 2: {player2_score} | Draws: {draws}")
            if player1_score > player2_score:
                print("Player 1 WINS! 🎉")
            elif player2_score > player1_score:
                print("Player 2 WINS! 🎉")
            else:
                print("It's a TIE! 🤝")


# =====================================
# Main entry point
# =====================================
def main(model_path, scaler_path, camera_width=1280, camera_height=720, countdown_duration=3):
    """
    Main function to run the game

    Args:
        model_path: Path to trained model
        scaler_path: Path to feature scaler
        camera_width: Width of camera capture
        camera_height: Height of camera capture
        countdown_duration: Duration of countdown in seconds
    """
    game = RPSGame(model_path, scaler_path, camera_width, camera_height, countdown_duration)
    game.run()


if __name__ == "__main__":
    # =====================================
    # Configuration parameters
    # =====================================
    MODEL_PATH = "model/rps_ridge_model.joblib"
    SCALER_PATH = "model/rps_scaler.joblib"
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    COUNTDOWN_DURATION = 3

    # Run the game
    main(MODEL_PATH, SCALER_PATH, CAMERA_WIDTH, CAMERA_HEIGHT, COUNTDOWN_DURATION)
