# ✋ Hand Gesture Recognition using Machine Learning

A robust multi-class hand gesture recognition system built using landmark-based features and classical machine learning models.  
The project focuses on preprocessing 3D hand landmark coordinates, handling class imbalance, and comparing multiple models to achieve high classification performance.

---

## 📌 Introduction

This project implements a **Hand Gesture Recognition System** using 21 hand landmarks per sample (each with x, y, z coordinates).

The objective is to classify hand gestures based on landmark coordinates while ensuring:

- Position invariance (gesture location in frame does not matter)
- Scale invariance (hand size or distance from camera does not matter)
- Proper handling of class imbalance
- Model comparison and selection based on performance metrics

After extensive experimentation, an **SVC model with polynomial kernel (C=30)** achieved the best performance.

---

## 📂 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset Description](#-dataset-description)
- [Data Preprocessing](#-data-preprocessing)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Data Splitting](#-data-splitting)
- [Model Training & Evaluation](#-model-training--evaluation)
  - [Random Forest](#-random-forest)
  - [Support Vector Classifier (SVC)](#-support-vector-classifier-svc)
  - [XGBoost](#-xgboost)
- [Model Comparison](#-model-comparison)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Dependencies](#-dependencies)
- [Configuration](#-configuration)
- [Examples](#-examples)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-future-improvements)
- [Contributors](#-contributors)
- [License](#-license)

---

## 📖 Project Overview

The pipeline follows these steps:

1. Import and verify dataset
2. Perform exploratory data analysis
3. Normalize landmark coordinates
4. Handle class imbalance
5. Stratified train/validation/test split
6. Train multiple models
7. Compare performance metrics
8. Select best-performing model

---

## 📊 Dataset Description

- Each sample contains **21 hand landmarks**
- Each landmark has **3 coordinates (x, y, z)**
- Total input features: **63 features per sample**
- Target column: `label`
- Multi-class classification problem

### ⚠ Class Imbalance

Some classes (e.g., *fist*, *mute*, *one*) are underrepresented.

To address this:
- Used **stratified splitting**
- Used `class_weight='balanced'` where supported

---

## 🔄 Data Preprocessing

### 1️⃣ Data Type Verification

- Verified dataframe shape
- Checked data types
- Performed descriptive statistics

---

### 2️⃣ Recentering (Translation Normalization)

- Wrist landmark was set as the origin (0,0,0)
- All other landmarks made relative to wrist

**Why?**

This ensures:
- Gesture position in the image frame does not affect classification
- Model focuses on gesture structure rather than placement

---

### 3️⃣ Scale Normalization

All landmarks were divided by **hand length**, calculated as:

> Euclidean distance between middle finger tip and wrist

**Why?**

This removes:
- Differences in hand size
- Distance from camera
- Perspective scaling effects

After normalization:
- Samples from the same class become more geometrically aligned
- Model focuses on gesture shape only

---

## 📈 Exploratory Data Analysis

- Visualized raw samples before normalization
- Visualized samples after normalization
- Countplot for class distribution
- Per-class sample visualization
- Confusion matrices for model evaluation

Normalization significantly improved alignment of same-class samples.

---

## 🔀 Data Splitting

Used:

```python
train_test_split(..., stratify=y)
```

This ensures:
- Class proportions remain consistent across train, validation, and test sets

---

# 🤖 Model Training & Evaluation

All models evaluated using:

- Accuracy
- F1-Score (weighted)
- Precision
- Recall
- Confusion Matrix

---

## 🌲 Random Forest

Experiments:

- 100 estimators
- 400 estimators (balanced)
- 600 estimators (balanced)

### Best RF Version:
- **100 estimators**
- Slightly better performance than 400/600
- Lower computational cost

Performance:
- Accuracy: 0.9759

---

## 🧠 Support Vector Classifier (SVC)

Tested multiple configurations:

| Kernel | C Value |
|--------|---------|
| RBF | 1.0 (default) |
| RBF | 0.1 |
| RBF | 10 |
| RBF | 30 |
| Poly | 30 |

### 🏆 Best SVC Model:
- Kernel: **Polynomial**
- C = **30**

Performance:
- Accuracy: **0.9860**
- F1-Score: **0.9860**
- Precision: **0.9861**
- Recall: **0.9860**

---

## 🚀 XGBoost

Experiments:

- 200 estimators
- 400 estimators
- 500 estimators

All models used:

```python
objective='multi:softprob'
eval_metric='mlogloss'
learning_rate=0.1
class_weight='balanced'
```

### Best XGBoost Version:
- 500 estimators (slightly better)
- 200 estimators chosen for testing (lighter with similar performance)

Performance (200 estimators):
- Accuracy: 0.9836

---

# 🏆 Model Comparison

| Model                    |  Accuracy  |  F1-Score  | Precision  |   Recall   |
| :----------------------- | :--------: | :--------: | :--------: | :--------: |
| **SVC (Poly, C=30)** 🥇  | **0.9860** | **0.9860** | **0.9861** | **0.9860** |
| XGBoost (200 est.)       |   0.9836   |   0.9836   |   0.9838   |   0.9836   |
| Random Forest (100 est.) |   0.9759   |   0.9758   |   0.9760   |   0.9759   |

### ✅ Final Selected Model:
**SVC with Polynomial Kernel (C=30)**

---

# ⚙ Installation

```bash
git clone https://github.com/your-username/hand-gesture-recognition.git
cd hand-gesture-recognition
pip install -r requirements.txt
```

---

# ▶ Usage

1. Ensure dataset is placed correctly:

```
data/hand_landmarks_data.csv
```

2. Run notebook:

```
ML_project.ipynb
```

3. Train models and evaluate performance.

---

# 📁 Project Structure

```
.
├── data/
│   └── hand_landmarks_data.csv
├── ML_project.ipynb
├── requirements.txt
├── README.md
```

---

# 🔧 Configuration

Key configurable components:

- Number of estimators (RF, XGBoost)
- SVC kernel & C value
- Train-test split ratio
- Learning rate (XGBoost)

---

# 🛠 Troubleshooting

### Issue: Poor performance
- Ensure normalization is applied correctly
- Confirm stratified splitting
- Check class imbalance handling

### Issue: XGBoost training slow
- Reduce number of estimators
- Lower learning rate cautiously

### Issue: Overfitting
- Reduce model complexity
- Use cross-validation
- Tune hyperparameters

---

# 🚀 Future Improvements

- Deep Learning models (LSTM / CNN on landmarks)
- Real-time webcam integration
- Hyperparameter tuning with GridSearchCV
- Cross-validation experiments
- Model deployment (Flask / FastAPI)
- Gesture-to-action system integration

---

# ⭐ Final Result

Through systematic preprocessing, normalization, and model comparison,  
the project achieved **98.60% accuracy** in multi-class hand gesture recognition using a classical ML pipeline.

The key success factor was **proper geometric normalization** of landmark data before training.