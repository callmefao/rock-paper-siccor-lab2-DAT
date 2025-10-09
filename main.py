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

# Import module chung
from hand_feature_extractor import HandFeatureExtractor


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
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # For drawing landmarks
        self.mp_hands = self.feature_extractor.mp_hands
        self.mp_drawing = self.feature_extractor.mp_drawing

        # Thread-safe data storage
        self.lock = threading.Lock()
        self.frame = None
        self.features = None
        self.landmarks = None
        self.normalized_landmarks = None
        self.prediction = None
        self.prediction_buffer = deque(maxlen=7)  # Smoothing buffer

        # Thread control
        self.running = False
        self.thread = None

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

    def update_frame(self, frame):
        """Update the frame to be processed (thread-safe)"""
        with self.lock:
            self.frame = frame.copy()

    def get_results(self):
        """Get the latest processing results (thread-safe)"""
        with self.lock:
            return {
                'features': self.features,
                'landmarks': self.landmarks,
                'normalized': self.normalized_landmarks,
                'prediction': self.prediction
            }

    def _process_loop(self):
        """Main processing loop running in separate thread"""
        while self.running:
            # Get frame to process
            with self.lock:
                if self.frame is None:
                    time.sleep(0.01)
                    continue
                frame_to_process = self.frame.copy()

            # Process frame
            features, landmarks, normalized = self._extract_landmarks(frame_to_process)

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
                self.normalized_landmarks = normalized
                self.prediction = smoothed_prediction

            time.sleep(0.01)  # Small delay to avoid CPU overload

    def _extract_landmarks(self, frame):
        """Extract landmarks from frame using HandFeatureExtractor"""
        # Extract landmarks array
        landmarks_array = self.feature_extractor.extract_landmarks_from_image(frame)

        if landmarks_array is None:
            return None, None, None

        # Store original for drawing (convert back to mediapipe format)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.feature_extractor.hands.process(image_rgb)
        original_landmarks = results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None

        # Normalize orientation
        normalized = normalize_hand_orientation(landmarks_array)

        # Extract features
        features = self.feature_extractor.extract_features_from_landmarks(normalized)

        return features, original_landmarks, normalized

    def _predict_gesture(self, features):
        """Predict gesture from features"""
        if features is None:
            return None

        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]

        labels = {0: "Rock", 1: "Paper", 2: "Scissors"}
        return labels[prediction]

    def _get_smoothed_prediction(self):
        """Get most common prediction from buffer"""
        if not self.prediction_buffer:
            return None

        # Count occurrences
        counts = {}
        for pred in self.prediction_buffer:
            if pred:
                counts[pred] = counts.get(pred, 0) + 1

        if not counts:
            return None

        # Return most common
        return max(counts, key=counts.get)


# =====================================
# Helper functions
# =====================================
def normalize_hand_orientation(landmarks):
    """
    Normalize hand orientation to match training data (hand pointing up)
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

    rotated_landmarks = landmarks.copy()
    for i in range(len(landmarks)):
        point_2d = landmarks[i][:2] - wrist[:2]
        rotated_point_2d = rotation_matrix_2d @ point_2d
        rotated_landmarks[i][:2] = rotated_point_2d + wrist[:2]
        rotated_landmarks[i][2] = landmarks[i][2]

    return rotated_landmarks


def draw_normalized_hand(frame, landmarks, position="top-left"):
    """Draw normalized hand visualization"""
    viz_size = 200
    viz_img = np.zeros((viz_size, viz_size, 3), dtype=np.uint8)

    landmarks_2d = landmarks[:, :2].copy()
    min_x, min_y = landmarks_2d.min(axis=0)
    max_x, max_y = landmarks_2d.max(axis=0)

    scale = 0.8 * viz_size / max(max_x - min_x, max_y - min_y)
    landmarks_2d = (landmarks_2d - [min_x, min_y]) * scale

    offset = [(viz_size - (max_x - min_x) * scale) / 2,
              (viz_size - (max_y - min_y) * scale) / 2]
    landmarks_2d = landmarks_2d + offset

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]

    for connection in connections:
        pt1 = tuple(landmarks_2d[connection[0]].astype(int))
        pt2 = tuple(landmarks_2d[connection[1]].astype(int))
        cv2.line(viz_img, pt1, pt2, (0, 255, 255), 2)

    for point in landmarks_2d:
        cv2.circle(viz_img, tuple(point.astype(int)), 3, (0, 255, 0), -1)

    cv2.putText(viz_img, "Normalized", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    h, w = frame.shape[:2]
    margin = 10

    if position == "top-left":
        x_offset, y_offset = margin, margin
    else:
        x_offset, y_offset = w - viz_size - margin, margin

    cv2.rectangle(frame, (x_offset-2, y_offset-2),
                  (x_offset+viz_size+2, y_offset+viz_size+2),
                  (255, 255, 255), 2)

    frame[y_offset:y_offset+viz_size, x_offset:x_offset+viz_size] = viz_img

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

    def run(self):
        """Run the main game loop"""
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

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
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                mid_width = width // 2

                # Split frame into two halves
                frame_left = frame[:, :mid_width].copy()
                frame_right = frame[:, mid_width:].copy()

                # Update frames for each player (threads will process them)
                player1.update_frame(frame_left)
                player2.update_frame(frame_right)

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
                    if results_p1['normalized'] is not None:
                        frame_left = draw_normalized_hand(frame_left, results_p1['normalized'], "top-left")

                if results_p2['landmarks']:
                    player2.mp_drawing.draw_landmarks(
                        frame_right, results_p2['landmarks'], player2.mp_hands.HAND_CONNECTIONS,
                        player2.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        player2.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )
                    if results_p2['normalized'] is not None:
                        frame_right = draw_normalized_hand(frame_right, results_p2['normalized'], "top-right")

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
                        player1_final = results_p1['prediction']
                        player2_final = results_p2['prediction']
                        winner = determine_winner(player1_final, player2_final)

                        if winner == "p1":
                            player1_score += 1
                            result = "Player 1 Wins!"
                        elif winner == "p2":
                            player2_score += 1
                            result = "Player 2 Wins!"
                        elif winner == "draw":
                            draws += 1
                            result = "Draw!"
                        else:
                            result = "No hands detected!"

                        result_time = time.time()
                        game_mode = "result"

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

                # Display scores
                score_text = f"P1: {player1_score}  |  Draws: {draws}  |  P2: {player2_score}"
                text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                score_x = (width - text_size[0]) // 2
                score_y = 50

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
                cv2.putText(combined_frame, "SPACE: Start | R: Reset | Q: Quit | Multi-threaded Mode",
                           (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow("Rock Paper Scissors - 2 Players (Multi-threaded)", combined_frame)

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
