# =====================================
# 2️⃣ Import thư viện
# =====================================
import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================
# 3️⃣ Khởi tạo Mediapipe Hands
# =====================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.3,  # Lower threshold to detect more hands
    min_tracking_confidence=0.3
)

# =====================================
# 4️⃣ Hàm trích xuất skeleton features (ENHANCED)
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

def extract_hand_landmarks(image_path):
    """Extract enhanced hand features including distances and angles"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]

    # Extract all landmarks
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append([lm.x, lm.y, lm.z])
    landmarks = np.array(landmarks)

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

    return enhanced_features

# =====================================
# 5️⃣ Data Augmentation
# =====================================
def augment_features(features, num_augmentations=3):
    """Augment features by adding small random noise"""
    augmented = [features]
    for _ in range(num_augmentations):
        noise = np.random.normal(0, 0.02, features.shape)  # Small Gaussian noise
        augmented.append(features + noise)
    return augmented

# =====================================
# 6️⃣ Load dataset và trích xuất đặc trưng
# =====================================
def load_dataset(folder_path, label, augment=False):
    X, y = [], []
    count = 0
    failed = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(folder_path, filename)
            features = extract_hand_landmarks(path)
            if features is not None:
                if augment:
                    # Apply data augmentation to training set
                    augmented_features = augment_features(features, num_augmentations=2)
                    for aug_feat in augmented_features:
                        X.append(aug_feat)
                        y.append(label)
                    count += len(augmented_features)
                else:
                    X.append(features)
                    y.append(label)
                    count += 1
            else:
                failed += 1

    print(f"   Loaded: {count} samples, Failed: {failed}")
    return X, y

train_root = "dataset/rps"
test_root = "dataset/rps-test-set"

labels = {"rock": 0, "paper": 1, "scissors": 2}

X_train, y_train, X_test, y_test = [], [], [], []

print("\n📂 Loading Training Data (with augmentation):")
for label_name, label_idx in labels.items():
    print(f"  {label_name}:")
    X1, y1 = load_dataset(os.path.join(train_root, label_name), label_idx, augment=True)
    X_train.extend(X1)
    y_train.extend(y1)

print("\n📂 Loading Test Data:")
for label_name, label_idx in labels.items():
    print(f"  {label_name}:")
    X2, y2 = load_dataset(os.path.join(test_root, label_name), label_idx, augment=False)
    X_test.extend(X2)
    y_test.extend(y2)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(f"\n✅ Total Train samples: {len(X_train)} | Test samples: {len(X_test)}")
print(f"   Feature dimension: {X_train.shape[1]}")

# =====================================
# 7️⃣ Chuẩn hóa dữ liệu
# =====================================
print("\n⚙️ Normalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================
# 8️⃣ Hyperparameter Tuning với GridSearchCV
# =====================================
print("\n🔍 Performing hyperparameter tuning (this may take a while)...")

param_grid = {
    'C': [1, 10, 100, 500],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}

grid_search = GridSearchCV(
    svm.SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\n✨ Best parameters found: {grid_search.best_params_}")
print(f"✨ Best cross-validation score: {grid_search.best_score_:.4f}")

clf = grid_search.best_estimator_

# =====================================
# 9️⃣ Cross-validation on training set
# =====================================
print("\n📊 Cross-validation scores:")
cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
print(f"   CV Scores: {cv_scores}")
print(f"   Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# =====================================
# 🔟 Đánh giá mô hình trên test set
# =====================================
print("\n🎯 Evaluating on test set...")
y_pred = clf.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\n📈 Test accuracy: {acc:.4f}\n")
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=labels.keys()))

# =====================================
# 1️⃣1️⃣ Confusion Matrix Visualization
# =====================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels.keys(),
            yticklabels=labels.keys())
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
os.makedirs("model", exist_ok=True)
plt.savefig('model/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("📊 Confusion matrix saved to model/confusion_matrix.png")

# =====================================
# 1️⃣2️⃣ Lưu mô hình và scaler
# =====================================
dump(clf, "model/rps_svm_model.joblib")
dump(scaler, "model/rps_scaler.joblib")
print("\n💾 Saved SVM model and scaler to model/")
print("🎉 Training complete!")
