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

# =====================================
# 1️⃣ Load trained model and scaler
# =====================================
clf = load("model/rps_ridge_model.joblib")
scaler = load("model/rps_scaler.joblib")

# =====================================
# 2️⃣ Player class with separate Mediapipe instance
# =====================================
class Player:
    """Each player has their own Mediapipe hands instance and processing thread"""

    def __init__(self, player_id, name):
        self.player_id = player_id
        self.name = name

        # Separate Mediapipe instance for this player
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

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
        self.hands.close()

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
        """Extract landmarks from frame"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None, None, None

        hand_landmarks = results.multi_hand_landmarks[0]

        # Extract all landmarks
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        landmarks = np.array(landmarks)

        # Store original
        original_landmarks = hand_landmarks

        # Normalize orientation
        normalized = normalize_hand_orientation(landmarks)

        # Extract features
        features = extract_features_from_landmarks(normalized)

        return features, original_landmarks, normalized

    def _predict_gesture(self, features):
        """Predict gesture from features"""
        if features is None:
            return None

        features_scaled = scaler.transform([features])
        prediction = clf.predict(features_scaled)[0]

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
# 3️⃣ Feature extraction functions (refactored)
# =====================================
def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def calculate_angle(p1, p2, p3):
    """Calculate angle at p2 formed by p1-p2-p3"""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)

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

def extract_features_from_landmarks(landmarks):
    """Extract enhanced features from landmarks"""
    # Basic features: normalized coordinates relative to wrist
    wrist = landmarks[0]
    normalized_landmarks = landmarks - wrist
    basic_features = normalized_landmarks.flatten()

    # Enhanced features: distances between key points
    distances = []
    fingertips = [4, 8, 12, 16, 20]

    for tip in fingertips:
        distances.append(calculate_distance(landmarks[tip], landmarks[0]))

    palm_center = np.mean(landmarks[[0, 5, 9, 13, 17]], axis=0)
    for tip in fingertips:
        distances.append(calculate_distance(landmarks[tip], palm_center))

    for i in range(len(fingertips) - 1):
        distances.append(calculate_distance(landmarks[fingertips[i]], landmarks[fingertips[i+1]]))

    # Finger angles
    angles = []
    finger_connections = [
        [2, 3, 4], [5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18, 19]
    ]

    for conn in finger_connections:
        angles.append(calculate_angle(landmarks[conn[0]], landmarks[conn[1]], landmarks[conn[2]]))

    palm_spread = calculate_distance(landmarks[2], landmarks[17])

    enhanced_features = np.concatenate([
        basic_features,
        np.array(distances),
        np.array(angles),
        [palm_spread]
    ])

    return enhanced_features

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

# =====================================
# 4️⃣ Game logic functions
# =====================================
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
# 5️⃣ Main game loop
# =====================================
def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Create two players with separate threads
    player1 = Player(1, "Player 1")
    player2 = Player(2, "Player 2")

    # Start processing threads
    player1.start()
    player2.start()

    # Game state
    game_mode = "play"
    countdown_start = None
    countdown_duration = 3
    player1_final = None
    player2_final = None
    result = ""
    result_time = None

    # Score tracking
    player1_score = 0
    player2_score = 0
    draws = 0

    print("🎮 Rock Paper Scissors Game Started! (Multi-threaded)")
    print("📹 Controls:")
    print("   SPACE - Start countdown and capture gestures")
    print("   R - Reset game and scores")
    print("   Q - Quit")
    print("\n🎯 Position your hands:")
    print("   Player 1: Left side of screen")
    print("   Player 2: Right side of screen")
    print("\n⚡ Using separate threads for each player - reduced interference!")

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
                remaining = countdown_duration - elapsed

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
                print("\n🔄 Scores reset!")

    finally:
        # Cleanup
        print("\n🛑 Stopping players...")
        player1.stop()
        player2.stop()
        cap.release()
        cv2.destroyAllWindows()

        # Print final scores
        print("\n" + "="*50)
        print("🏆 FINAL SCORES 🏆")
        print("="*50)
        print(f"Player 1: {player1_score}")
        print(f"Player 2: {player2_score}")
        print(f"Draws: {draws}")
        if player1_score > player2_score:
            print("\n🎉 Player 1 is the WINNER! 🎉")
        elif player2_score > player1_score:
            print("\n🎉 Player 2 is the WINNER! 🎉")
        else:
            print("\n🤝 It's a TIE! 🤝")
        print("="*50)

if __name__ == "__main__":
    main()