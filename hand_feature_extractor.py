"""
Hand Feature Extractor Module
Shared module for extracting hand landmarks and features using MediaPipe
Used by both training and inference
"""

import cv2
import numpy as np
import mediapipe as mp


class HandFeatureExtractor:
    """Class for extracting hand landmarks and features using MediaPipe"""

    def __init__(self, static_image_mode=True, max_num_hands=1,
                 min_detection_confidence=0.3, min_tracking_confidence=0.3):
        """
        Initialize MediaPipe Hands

        Args:
            static_image_mode: Whether to treat input as static images
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def close(self):
        """Close MediaPipe Hands"""
        self.hands.close()

    @staticmethod
    def calculate_distance(p1, p2):
        """Calculate Euclidean distance between two 3D points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

    @staticmethod
    def calculate_angle(p1, p2, p3):
        """
        Calculate angle at p2 formed by p1-p2-p3

        Args:
            p1, p2, p3: 3D points

        Returns:
            Angle in radians
        """
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]])

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)

    def extract_landmarks_from_image(self, image):
        """
        Extract hand landmarks from image array

        Args:
            image: BGR image array

        Returns:
            numpy array of shape (21, 3) containing [x, y, z] coordinates,
            or None if no hand detected
        """
        if image is None:
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None

        hand_landmarks = results.multi_hand_landmarks[0]

        # Extract all landmarks
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        return np.array(landmarks)

    def extract_features_from_landmarks(self, landmarks):
        """
        Extract enhanced features from landmarks

        Args:
            landmarks: numpy array of shape (21, 3) containing hand landmarks

        Returns:
            1D numpy array containing all features
        """
        if landmarks is None:
            return None

        # Basic features: normalized coordinates relative to wrist
        wrist = landmarks[0]
        normalized_landmarks = landmarks - wrist
        basic_features = normalized_landmarks.flatten()

        # Enhanced features: distances between key points
        distances = []
        fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips

        # Fingertip to wrist distances
        for tip in fingertips:
            distances.append(self.calculate_distance(landmarks[tip], landmarks[0]))

        # Fingertip to palm center distances
        palm_center = np.mean(landmarks[[0, 5, 9, 13, 17]], axis=0)
        for tip in fingertips:
            distances.append(self.calculate_distance(landmarks[tip], palm_center))

        # Distances between consecutive fingertips
        for i in range(len(fingertips) - 1):
            distances.append(self.calculate_distance(landmarks[fingertips[i]], landmarks[fingertips[i+1]]))

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
            angles.append(self.calculate_angle(landmarks[conn[0]], landmarks[conn[1]], landmarks[conn[2]]))

        # Palm spread (distance between thumb base and pinky base)
        palm_spread = self.calculate_distance(landmarks[2], landmarks[17])

        # Combine all features
        enhanced_features = np.concatenate([
            basic_features,
            np.array(distances),
            np.array(angles),
            [palm_spread]
        ])

        return enhanced_features

    def extract_features_from_image(self, image):
        """
        Extract features directly from image

        Args:
            image: BGR image array

        Returns:
            1D numpy array containing all features, or None if no hand detected
        """
        landmarks = self.extract_landmarks_from_image(image)
        if landmarks is None:
            return None
        return self.extract_features_from_landmarks(landmarks)

    def extract_features_from_file(self, image_path):
        """
        Extract features from image file

        Args:
            image_path: Path to image file

        Returns:
            1D numpy array containing all features, or None if no hand detected
        """
        image = cv2.imread(image_path)
        return self.extract_features_from_image(image)


class ImageAugmentor:
    """Class for augmenting images for training"""

    @staticmethod
    def augment_image(image):
        """
        Augment image with rotation and flipping

        Args:
            image: BGR image array

        Returns:
            List of augmented images including original
        """
        augmented_images = [image]

        # Horizontal flip (mirror)
        flipped_h = cv2.flip(image, 1)
        augmented_images.append(flipped_h)

        # Rotate -15 and +15 degrees
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        for angle in [-15, 15]:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            augmented_images.append(rotated)

        # Flip + rotate combination
        M_rot = cv2.getRotationMatrix2D(center, 10, 1.0)
        flipped_rotated = cv2.warpAffine(flipped_h, M_rot, (w, h), borderMode=cv2.BORDER_REPLICATE)
        augmented_images.append(flipped_rotated)

        return augmented_images

    @staticmethod
    def augment_features(features, num_augmentations=2):
        """
        Augment features by adding small random noise

        Args:
            features: 1D feature array
            num_augmentations: Number of augmented versions to create

        Returns:
            List of augmented feature arrays including original
        """
        augmented = [features]
        for _ in range(num_augmentations):
            noise = np.random.normal(0, 0.015, features.shape)
            augmented.append(features + noise)
        return augmented

