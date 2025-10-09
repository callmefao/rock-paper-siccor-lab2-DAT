"""
Rock Paper Scissors - Model Training Module
Train a classifier to recognize rock, paper, scissors gestures
"""

# =====================================
# Import thư viện
# =====================================
import os
import cv2
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns

# Import module chung
from hand_feature_extractor import HandFeatureExtractor, ImageAugmentor


# =====================================
# Dataset Loader Class
# =====================================
class RPSDatasetLoader:
    """Class for loading and augmenting Rock Paper Scissors dataset"""

    def __init__(self, feature_extractor, image_augmentor=None):
        """
        Initialize dataset loader

        Args:
            feature_extractor: HandFeatureExtractor instance
            image_augmentor: ImageAugmentor instance for data augmentation
        """
        self.feature_extractor = feature_extractor
        self.image_augmentor = image_augmentor
        self.labels = {"rock": 0, "paper": 1, "scissors": 2}

    def load_dataset(self, folder_path, label, augment=False):
        """
        Load dataset from folder

        Args:
            folder_path: Path to folder containing images
            label: Label index for this class
            augment: Whether to apply data augmentation

        Returns:
            Tuple of (features_list, labels_list)
        """
        X, y = [], []
        count = 0
        failed = 0

        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                path = os.path.join(folder_path, filename)

                if augment and self.image_augmentor:
                    # Load image for augmentation
                    image = cv2.imread(path)
                    if image is None:
                        failed += 1
                        continue

                    # Apply image augmentation (rotate, flip)
                    augmented_images = self.image_augmentor.augment_image(image)

                    for aug_img in augmented_images:
                        features = self.feature_extractor.extract_features_from_image(aug_img)
                        if features is not None:
                            # Apply feature-level noise augmentation
                            feature_augmented = self.image_augmentor.augment_features(features, num_augmentations=1)
                            for aug_feat in feature_augmented:
                                X.append(aug_feat)
                                y.append(label)
                            count += len(feature_augmented)
                        else:
                            failed += 1
                else:
                    # No augmentation
                    features = self.feature_extractor.extract_features_from_file(path)
                    if features is not None:
                        X.append(features)
                        y.append(label)
                        count += 1
                    else:
                        failed += 1

        print(f"   Loaded: {count} samples, Failed: {failed}")
        return X, y

    def load_train_test_split(self, train_root, test_root, augment_train=True):
        """
        Load both training and test datasets

        Args:
            train_root: Root folder for training data
            test_root: Root folder for test data
            augment_train: Whether to augment training data

        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        X_train, y_train, X_test, y_test = [], [], [], []

        print("\n📂 Loading Training Data" + (" (augmented)" if augment_train else ""))
        for label_name, label_idx in self.labels.items():
            print(f"  {label_name}:", end=" ")
            X1, y1 = self.load_dataset(
                os.path.join(train_root, label_name),
                label_idx,
                augment=augment_train
            )
            X_train.extend(X1)
            y_train.extend(y1)

        print("\n📂 Loading Test Data")
        for label_name, label_idx in self.labels.items():
            print(f"  {label_name}:", end=" ")
            X2, y2 = self.load_dataset(
                os.path.join(test_root, label_name),
                label_idx,
                augment=False
            )
            X_test.extend(X2)
            y_test.extend(y2)

        return (np.array(X_train), np.array(y_train),
                np.array(X_test), np.array(y_test))


# =====================================
# Model Trainer Class
# =====================================
class RPSModelTrainer:
    """Class for training and evaluating Rock Paper Scissors classifier"""

    def __init__(self, model=None, scaler=None):
        """
        Initialize trainer

        Args:
            model: Sklearn classifier (default: RidgeClassifier)
            scaler: Feature scaler (default: StandardScaler)
        """
        self.model = model if model else RidgeClassifier(random_state=42)
        self.scaler = scaler if scaler else StandardScaler()
        self.labels = {0: "rock", 1: "paper", 2: "scissors"}
        self.best_model = None

    def normalize_data(self, X_train, X_test):
        """
        Normalize features using StandardScaler

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        print("\n⚙️ Normalizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def hyperparameter_tuning(self, X_train, y_train, param_grid=None):
        """
        Perform hyperparameter tuning using GridSearchCV

        Args:
            X_train: Training features (normalized)
            y_train: Training labels
            param_grid: Grid of parameters to search (optional)

        Returns:
            Best estimator found
        """
        if param_grid is None:
            param_grid = {
                'alpha': [0.1, 1.0, 10.0, 100.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
            }

        print("\n🔍 Performing hyperparameter tuning (this may take a while)...")

        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"\n✨ Best parameters found: {grid_search.best_params_}")
        print(f"✨ Best cross-validation score: {grid_search.best_score_:.4f}")

        self.best_model = grid_search.best_estimator_
        return self.best_model

    def cross_validate(self, X_train, y_train, cv=5):
        """
        Perform cross-validation

        Args:
            X_train: Training features (normalized)
            y_train: Training labels
            cv: Number of folds

        Returns:
            Cross-validation scores
        """
        print("\n📊 Cross-validation scores:")
        model = self.best_model if self.best_model else self.model
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
        print(f"   CV Scores: {cv_scores}")
        print(f"   Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        return cv_scores

    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set

        Args:
            X_test: Test features (normalized)
            y_test: Test labels

        Returns:
            Tuple of (accuracy, predictions)
        """
        print("\n🎯 Evaluating on test set...")
        model = self.best_model if self.best_model else self.model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n📈 Test accuracy: {acc:.4f}\n")
        print("Classification report:")
        print(classification_report(y_test, y_pred, target_names=self.labels.values()))

        return acc, y_pred

    def plot_confusion_matrix(self, y_test, y_pred, save_path='model/confusion_matrix.png'):
        """
        Plot and save confusion matrix

        Args:
            y_test: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.labels.values(),
                    yticklabels=self.labels.values())
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to {save_path}")
        plt.close()

    def save_model(self, model_path='model/rps_ridge_model.joblib',
                   scaler_path='model/rps_scaler.joblib'):
        """
        Save trained model and scaler

        Args:
            model_path: Path to save model
            scaler_path: Path to save scaler
        """
        model = self.best_model if self.best_model else self.model

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        dump(model, model_path)
        dump(self.scaler, scaler_path)

        print(f"\n💾 Saved model to {model_path}")
        print(f"💾 Saved scaler to {scaler_path}")


# =====================================
# Main Training Pipeline
# =====================================
class TrainingPipeline:
    """Complete training pipeline for RPS classifier"""

    def __init__(self, train_root, test_root,
                 static_image_mode=True, max_num_hands=1,
                 min_detection_confidence=0.3, min_tracking_confidence=0.3):
        """
        Initialize training pipeline

        Args:
            train_root: Root folder for training data
            test_root: Root folder for test data
            static_image_mode: Whether to treat input as static images
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.train_root = train_root
        self.test_root = test_root

        # Initialize components
        self.feature_extractor = HandFeatureExtractor(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.image_augmentor = ImageAugmentor()
        self.dataset_loader = RPSDatasetLoader(self.feature_extractor, self.image_augmentor)
        self.trainer = RPSModelTrainer()

    def run(self, augment_train=True, tune_hyperparameters=True,
            param_grid=None, model_path='model/rps_ridge_model.joblib',
            scaler_path='model/rps_scaler.joblib',
            confusion_matrix_path='model/confusion_matrix.png'):
        """
        Run complete training pipeline

        Args:
            augment_train: Whether to augment training data
            tune_hyperparameters: Whether to perform hyperparameter tuning
            param_grid: Grid of parameters for hyperparameter tuning
            model_path: Path to save trained model
            scaler_path: Path to save scaler
            confusion_matrix_path: Path to save confusion matrix plot
        """
        print("=" * 60)
        print("🎮 Rock Paper Scissors - Model Training Pipeline")
        print("=" * 60)

        # Load data
        X_train, y_train, X_test, y_test = self.dataset_loader.load_train_test_split(
            self.train_root, self.test_root, augment_train=augment_train
        )

        print(f"\n✅ Total Train samples: {len(X_train)} | Test samples: {len(X_test)}")
        print(f"   Feature dimension: {X_train.shape[1]}")

        # Normalize data
        X_train_scaled, X_test_scaled = self.trainer.normalize_data(X_train, X_test)

        # Train model
        if tune_hyperparameters:
            self.trainer.hyperparameter_tuning(X_train_scaled, y_train, param_grid)
        else:
            print("\n🔧 Training model with default parameters...")
            self.trainer.model.fit(X_train_scaled, y_train)

        # Cross-validation
        self.trainer.cross_validate(X_train_scaled, y_train)

        # Evaluate
        acc, y_pred = self.trainer.evaluate(X_test_scaled, y_test)

        # Plot confusion matrix
        self.trainer.plot_confusion_matrix(y_test, y_pred, confusion_matrix_path)

        # Save model
        self.trainer.save_model(model_path, scaler_path)

        # Cleanup
        self.feature_extractor.close()

        print("\n🎉 Training complete!")
        print("=" * 60)

        return acc


# =====================================
# Main entry point
# =====================================
def main(train_root, test_root, augment_train=True, tune_hyperparameters=True,
         param_grid=None, model_path='model/rps_ridge_model.joblib',
         scaler_path='model/rps_scaler.joblib',
         confusion_matrix_path='model/confusion_matrix.png',
         static_image_mode=True, max_num_hands=1,
         min_detection_confidence=0.3, min_tracking_confidence=0.3):
    """
    Main function to run training pipeline

    Args:
        train_root: Root folder for training data
        test_root: Root folder for test data
        augment_train: Whether to augment training data
        tune_hyperparameters: Whether to perform hyperparameter tuning
        param_grid: Grid of parameters for hyperparameter tuning
        model_path: Path to save trained model
        scaler_path: Path to save scaler
        confusion_matrix_path: Path to save confusion matrix plot
        static_image_mode: Whether to treat input as static images
        max_num_hands: Maximum number of hands to detect
        min_detection_confidence: Minimum confidence for hand detection
        min_tracking_confidence: Minimum confidence for hand tracking
    """
    pipeline = TrainingPipeline(
        train_root, test_root,
        static_image_mode, max_num_hands,
        min_detection_confidence, min_tracking_confidence
    )
    pipeline.run(
        augment_train=augment_train,
        tune_hyperparameters=tune_hyperparameters,
        param_grid=param_grid,
        model_path=model_path,
        scaler_path=scaler_path,
        confusion_matrix_path=confusion_matrix_path
    )


if __name__ == "__main__":
    # =====================================
    # Configuration parameters
    # =====================================

    # Dataset paths
    TRAIN_ROOT = "dataset/rps"
    TEST_ROOT = "dataset/rps-test-set"

    # Model save paths
    MODEL_PATH = "model/rps_ridge_model.joblib"
    SCALER_PATH = "model/rps_scaler.joblib"
    CONFUSION_MATRIX_PATH = "model/confusion_matrix.png"

    # Training options
    AUGMENT_TRAIN = True
    TUNE_HYPERPARAMETERS = True

    # Hyperparameter tuning grid
    PARAM_GRID = {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }

    # MediaPipe configuration
    STATIC_IMAGE_MODE = True
    MAX_NUM_HANDS = 1
    MIN_DETECTION_CONFIDENCE = 0.3
    MIN_TRACKING_CONFIDENCE = 0.3

    # Run training
    main(
        train_root=TRAIN_ROOT,
        test_root=TEST_ROOT,
        augment_train=AUGMENT_TRAIN,
        tune_hyperparameters=TUNE_HYPERPARAMETERS,
        param_grid=PARAM_GRID,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        confusion_matrix_path=CONFUSION_MATRIX_PATH,
        static_image_mode=STATIC_IMAGE_MODE,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )
