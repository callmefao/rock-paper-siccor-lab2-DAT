# 🎮 Rock Paper Scissors - Real-time Game

A real-time rock-paper-scissors game using **MediaPipe** for hand detection and **SVM** for gesture classification.

## ✨ Features

- **2-Player Split Screen** with real-time hand gesture recognition  
- **Hand Orientation Normalization** – works at any angle  
- **Visual Feedback** – shows normalized hand and skeleton  

## 🎯 Demo

| Rock vs Scissors | Rock vs Paper | Scissors vs Paper |
|:---:|:---:|:---:|
| ![](./demo/rock-scissor.png) | ![](./demo/rock-paper.png) | ![](./demo/scissor-paper.png) |

## 🚀 Quick Start

**Install dependencies:**
```bash
pip install opencv-python mediapipe scikit-learn joblib numpy matplotlib seaborn
```

**Train model:**
```bash
python train.py
```

**Play game:**
```bash
python main.py
```

## 🎮 Controls

- **SPACE** - Start countdown and play
- **R** - Reset scores
- **Q** - Quit

## 🔧 Tech Stack

- **Hand Detection**: MediaPipe
- **Classification**: RidgeClassifier 
- **Features**: 83 features (landmarks, distances, angles, palm spread)
- **Data Augmentation**: 3x training samples
- **Test Accuracy**: ~1.00%

## 📊 Model Performance

| Class    | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Rock     | 1.00      | 1.00   | 1.00     |
| Paper    | 1.00      | 1.00   | 1.00     |
| Scissors | 1.00      | 1.00   | 1.00     |

---


**Overall Accuracy:** **100%**  
**Best Params:** `alpha=100`, `solver='auto'`  

Confusion matrix:  
![](./model/confusion_matrix.png)
