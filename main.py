# =====================================
# Rock Paper Scissors - 2 Player Game
# =====================================
import cv2
import numpy as np
import mediapipe as mp
from joblib import load
import time

# =====================================
# 1️⃣ Load trained model and scaler
# =====================================
clf = load("model/rps_svm_model.joblib")
scaler = load("model/rps_scaler.joblib")

# =====================================
# 2️⃣ Initialize Mediapipe Hands
# =====================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =====================================
# 3️⃣ Enhanced feature extraction functions
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
    This rotates the hand so that wrist-to-middle-finger direction points upward
    """
    # Key points: wrist (0) and middle finger MCP (9)
    wrist = landmarks[0]
    middle_mcp = landmarks[9]

    # Calculate current hand direction vector (wrist to middle finger base)
    hand_vector = middle_mcp - wrist
    hand_vector_2d = hand_vector[:2]  # Use only x, y for rotation

    # Calculate angle to rotate hand to point upward (negative y direction)
    # In image coordinates, up is negative y
    current_angle = np.arctan2(hand_vector_2d[0], -hand_vector_2d[1])

    # Create 2D rotation matrix
    cos_angle = np.cos(-current_angle)
    sin_angle = np.sin(-current_angle)
    rotation_matrix_2d = np.array([
        [cos_angle, -sin_angle],
        [sin_angle, cos_angle]
    ])

    # Rotate all landmarks (only x, y coordinates)
    rotated_landmarks = landmarks.copy()
    for i in range(len(landmarks)):
        # Translate to origin (relative to wrist)
        point_2d = landmarks[i][:2] - wrist[:2]
        # Rotate
        rotated_point_2d = rotation_matrix_2d @ point_2d
        # Translate back
        rotated_landmarks[i][:2] = rotated_point_2d + wrist[:2]
        # Keep z coordinate unchanged
        rotated_landmarks[i][2] = landmarks[i][2]

    return rotated_landmarks

def draw_normalized_hand(frame, landmarks, position="top-left"):
    """
    Draw normalized hand visualization in a small window
    position: "top-left" or "top-right"
    """
    # Create small visualization window
    viz_size = 200
    viz_img = np.zeros((viz_size, viz_size, 3), dtype=np.uint8)

    # Scale and center the landmarks for visualization
    landmarks_2d = landmarks[:, :2].copy()

    # Normalize to fit in visualization window
    min_x, min_y = landmarks_2d.min(axis=0)
    max_x, max_y = landmarks_2d.max(axis=0)

    # Scale to fit 80% of window
    scale = 0.8 * viz_size / max(max_x - min_x, max_y - min_y)
    landmarks_2d = (landmarks_2d - [min_x, min_y]) * scale

    # Center in window
    offset = [(viz_size - (max_x - min_x) * scale) / 2,
              (viz_size - (max_y - min_y) * scale) / 2]
    landmarks_2d = landmarks_2d + offset

    # Draw hand connections
    connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm
        (5, 9), (9, 13), (13, 17)
    ]

    for connection in connections:
        pt1 = tuple(landmarks_2d[connection[0]].astype(int))
        pt2 = tuple(landmarks_2d[connection[1]].astype(int))
        cv2.line(viz_img, pt1, pt2, (0, 255, 255), 2)

    # Draw landmarks
    for point in landmarks_2d:
        cv2.circle(viz_img, tuple(point.astype(int)), 3, (0, 255, 0), -1)

    # Add title
    cv2.putText(viz_img, "Normalized", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Overlay on main frame
    h, w = frame.shape[:2]
    margin = 10

    if position == "top-left":
        x_offset, y_offset = margin, margin
    else:  # top-right
        x_offset, y_offset = w - viz_size - margin, margin

    # Create border
    cv2.rectangle(frame, (x_offset-2, y_offset-2),
                  (x_offset+viz_size+2, y_offset+viz_size+2),
                  (255, 255, 255), 2)

    # Overlay visualization
    frame[y_offset:y_offset+viz_size, x_offset:x_offset+viz_size] = viz_img

    return frame

def extract_landmarks_from_frame(frame):
    """Extract enhanced hand landmarks from a frame region"""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None, None, None

    hand_landmarks = results.multi_hand_landmarks[0]

    # Extract all landmarks
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    landmarks = np.array(landmarks)

    # Store original landmarks for drawing
    original_landmarks = landmarks.copy()

    # **PREPROCESSING: Normalize hand orientation (rotate to point upward)**
    landmarks = normalize_hand_orientation(landmarks)

    # Basic features: normalized coordinates relative to wrist
    wrist = landmarks[0]
    normalized_landmarks = landmarks - wrist
    basic_features = normalized_landmarks.flatten()

    # Enhanced features: distances between key points
    distances = []
    # Fingertip to wrist distances
    fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips
    for tip in fingertips:
        distances.append(calculate_distance(landmarks[tip], landmarks[0]))

    # Fingertip to palm center distances
    palm_center = np.mean(landmarks[[0, 5, 9, 13, 17]], axis=0)
    for tip in fingertips:
        distances.append(calculate_distance(landmarks[tip], palm_center))

    # Distances between consecutive fingertips
    for i in range(len(fingertips) - 1):
        distances.append(calculate_distance(landmarks[fingertips[i]], landmarks[fingertips[i+1]]))

    # Finger angles (at MCP joints)
    angles = []
    finger_connections = [
        [2, 3, 4],    # Thumb
        [5, 6, 7],    # Index
        [9, 10, 11],  # Middle
        [13, 14, 15], # Ring
        [17, 18, 19]  # Pinky
    ]

    for conn in finger_connections:
        angles.append(calculate_angle(landmarks[conn[0]], landmarks[conn[1]], landmarks[conn[2]]))

    # Palm spread (distance between thumb base and pinky base)
    palm_spread = calculate_distance(landmarks[2], landmarks[17])

    # Combine all features
    enhanced_features = np.concatenate([
        basic_features,
        np.array(distances),
        np.array(angles),
        [palm_spread]
    ])

    return enhanced_features, hand_landmarks, landmarks

# =====================================
# 4️⃣ Predict gesture
# =====================================
def predict_gesture(features):
    """Predict rock, paper, or scissors from features"""
    if features is None:
        return None

    features_scaled = scaler.transform([features])
    prediction = clf.predict(features_scaled)[0]

    labels = {0: "Rock", 1: "Paper", 2: "Scissors"}
    return labels[prediction]

# =====================================
# 5️⃣ Determine winner
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
# 6️⃣ Main game loop
# =====================================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Game state
    game_mode = "play"  # "play" or "countdown"
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

    print("🎮 Rock Paper Scissors Game Started!")
    print("📹 Controls:")
    print("   SPACE - Start countdown and capture gestures")
    print("   R - Reset game and scores")
    print("   Q - Quit")
    print("\n🎯 Position your hands:")
    print("   Player 1: Left side of screen")
    print("   Player 2: Right side of screen")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror the frame
        height, width = frame.shape[:2]
        mid_width = width // 2

        # Split frame into two halves
        frame_left = frame[:, :mid_width].copy()
        frame_right = frame[:, mid_width:].copy()

        # Process both players
        features_p1, landmarks_p1, normalized_p1 = extract_landmarks_from_frame(frame_left)
        features_p2, landmarks_p2, normalized_p2 = extract_landmarks_from_frame(frame_right)

        # Draw hand landmarks (original)
        if landmarks_p1:
            mp_drawing.draw_landmarks(
                frame_left, landmarks_p1, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
            )
            # Draw normalized hand visualization
            if normalized_p1 is not None:
                frame_left = draw_normalized_hand(frame_left, normalized_p1, "top-left")

        if landmarks_p2:
            mp_drawing.draw_landmarks(
                frame_right, landmarks_p2, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
            )
            # Draw normalized hand visualization
            if normalized_p2 is not None:
                frame_right = draw_normalized_hand(frame_right, normalized_p2, "top-right")

        # Game logic
        if game_mode == "play":
            # Real-time prediction
            gesture_p1 = predict_gesture(features_p1) if features_p1 is not None else "No hand"
            gesture_p2 = predict_gesture(features_p2) if features_p2 is not None else "No hand"

            # Display gestures
            cv2.putText(frame_left, f"Player 1", (10, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_left, f"{gesture_p1}", (10, 290),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame_right, f"Player 2", (10, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_right, f"{gesture_p2}", (10, 290),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        elif game_mode == "countdown":
            # Countdown mode
            elapsed = time.time() - countdown_start
            remaining = countdown_duration - elapsed

            if remaining > 0:
                countdown_text = str(int(remaining) + 1)

                # Display countdown on both sides
                cv2.putText(frame_left, countdown_text, (mid_width//2 - 50, height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
                cv2.putText(frame_right, countdown_text, (mid_width//2 - 50, height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 8)
            else:
                # Capture final gestures
                player1_final = predict_gesture(features_p1) if features_p1 is not None else None
                player2_final = predict_gesture(features_p2) if features_p2 is not None else None
                winner = determine_winner(player1_final, player2_final)

                # Update scores
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
            # Display results
            cv2.putText(frame_left, f"Player 1", (10, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_left, f"{player1_final if player1_final else 'No hand'}",
                       (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame_right, f"Player 2", (10, 250),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_right, f"{player2_final if player2_final else 'No hand'}",
                       (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show result for 3 seconds, then return to play mode
            if time.time() - result_time > 3:
                game_mode = "play"
                player1_final = None
                player2_final = None
                result = ""

        # Combine frames
        combined_frame = np.hstack([frame_left, frame_right])

        # Draw center line
        cv2.line(combined_frame, (mid_width, 0), (mid_width, height), (255, 255, 255), 2)

        # Display scores at the top center
        score_text = f"P1: {player1_score}  |  Draws: {draws}  |  P2: {player2_score}"
        text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        score_x = (width - text_size[0]) // 2
        score_y = 50

        # Background for score
        cv2.rectangle(combined_frame,
                     (score_x - 10, score_y - text_size[1] - 10),
                     (score_x + text_size[0] + 10, score_y + 10),
                     (0, 0, 0), -1)

        # Score text
        cv2.putText(combined_frame, score_text, (score_x, score_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Highlight who's winning
        if player1_score > player2_score:
            cv2.putText(combined_frame, "WINNING!", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif player2_score > player1_score:
            cv2.putText(combined_frame, "WINNING!", (width - 250, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display result in center
        if result:
            text_size = cv2.getTextSize(result, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            text_x = (width - text_size[0]) // 2
            text_y = height - 50

            # Background rectangle
            cv2.rectangle(combined_frame,
                         (text_x - 10, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 0), -1)

            # Result text
            if "Player 1" in result:
                color = (0, 255, 255)
            elif "Player 2" in result:
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            cv2.putText(combined_frame, result, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)

        # Display instructions
        cv2.putText(combined_frame, "SPACE: Start | R: Reset Scores | Q: Quit",
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Rock Paper Scissors - 2 Players", combined_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' ') and game_mode == "play":
            # Start countdown
            game_mode = "countdown"
            countdown_start = time.time()
            result = ""
        elif key == ord('r'):
            # Reset game and scores
            game_mode = "play"
            player1_final = None
            player2_final = None
            result = ""
            player1_score = 0
            player2_score = 0
            draws = 0
            print("\n🔄 Scores reset!")

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
