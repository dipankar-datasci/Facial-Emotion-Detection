# 🧠 FaceRead — Facial Emotion Detection using FER-2013

Facial Emotion Detection bridges the gap between human expression and machine understanding. This project trains a custom Convolutional Neural Network (CNN) from scratch to classify facial expressions into **7 distinct emotions** using the FER-2013 benchmark dataset.

---

## 📌 What This Project Does

| | |
|---|---|
| **Goal** | Recognize human emotions from grayscale facial images |
| **Approach** | Custom CNN with data augmentation and class balancing |
| **Evaluation** | Accuracy, Confusion Matrix, F1-Score, ROC-AUC |
| **Demo** | Real-time webcam prediction via Google Colab |

**Emotion Classes:**
`Angry` · `Disgust` · `Fear` · `Happy` · `Sad` · `Surprise` · `Neutral`

---

## 🗂️ Dataset

**FER-2013 — Facial Expression Recognition Challenge**
- 📦 Source: [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- 🖼️ Format: 48 × 48 pixel grayscale images
- 🏷️ Classes: 7 emotion categories

**Dataset Split:**

| Split | Ratio |
|---|---|
| Training | 90% |
| Validation | 5% |
| Test | 5% |

**Augmentation applied to training data only:**
- Random Horizontal Flip
- Random Rotation (±15°)
- Random Crop → Resize back to 48 × 48
- Gaussian Blur
- Color Jitter (brightness, contrast)
- Random Perspective Distortion

**Class Balancing:** Underrepresented classes are oversampled with augmented copies to reach ~14,285 samples per class (~100,000 total).

---

## 🧠 Model Architecture — `FixedEmotionCNN`

A fully custom CNN — no pretrained backbone, trained end-to-end on FER-2013.

### Feature Extraction

```
Input: (Batch, 1, 48, 48)

Block 1 → Conv2d(1→64) × 2   |  BatchNorm → ReLU → MaxPool → Dropout(0.25)
Block 2 → Conv2d(64→128) × 2  |  BatchNorm → ReLU → MaxPool → Dropout(0.25)
Block 3 → Conv2d(128→256) × 2 |  BatchNorm → ReLU → MaxPool → Dropout(0.25)
```

### Classifier Head

```
Flatten → Linear(9216→512) → BatchNorm → ReLU → Dropout(0.5)
        → Linear(512→7)
```

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Input Size | 1 × 48 × 48 |
| Batch Size | 64 |
| Optimizer | Adam (lr=0.001, weight_decay=1e-4) |
| Loss Function | CrossEntropyLoss |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Early Stopping | Patience = 8 epochs |
| Max Epochs | 40 |

The best checkpoint (lowest validation loss) is saved automatically to `best_model.pth`.

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install torch torchvision torchinfo scikit-learn matplotlib seaborn opencv-python-headless Pillow
```

### 2. Run in Google Colab

1. Upload `CV_FINAL_PROJECT_DATASET.zip` to `/content/` — or mount Google Drive and update the path.
2. Open the notebook and run all cells in order.
3. `best_model.pth` is saved automatically after training completes.

### 3. Load a Pretrained Model

```python
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FixedEmotionCNN(num_classes=7)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()
```

---

## 📸 Live Demo (Google Colab)

The notebook includes a real-time webcam prediction flow:

1. 📷 Capture a photo via browser
2. 🔍 Detect and crop the face using OpenCV Haar cascades
3. 🧠 Run the face through the trained model
4. 🏷️ Display the predicted emotion

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Overall correct predictions across all classes |
| Confusion Matrix | Class-wise true vs. predicted breakdown |
| Precision / Recall / F1 | Per-class classification report |
| ROC Curve + AUC | One-vs-rest multi-class ROC analysis |

### Visualizations Included
- ✅ Training & Validation Loss / Accuracy curves
- ✅ Confusion Matrix (heatmap)
- ✅ Multi-Class ROC Curves
- ✅ Class distribution charts (before and after augmentation)
- ✅ Sample images from the dataset

---

## 📁 Repository Structure

```
FaceRead/
├── FaceRead_AI_Powered_Facial_Emotion_Detection.ipynb   # Full pipeline notebook
├── best_model.pth                                        # Saved weights (post-training)
├── CV_FINAL_PROJECT_DATASET.zip                         # FER-2013 dataset archive
└── README.md
```

---

## 🏁 Conclusion

FaceRead demonstrates that a well-designed custom CNN — trained with thoughtful augmentation and class balancing — can achieve strong performance on the FER-2013 benchmark without relying on any pretrained backbone. The full pipeline, from raw dataset to live webcam prediction, is contained in a single notebook.

---

## 📦 Requirements

| Library | Purpose |
|---|---|
| `torch` / `torchvision` | Model training & transforms |
| `torchinfo` | Model summary |
| `scikit-learn` | Metrics & stratified splits |
| `opencv-python-headless` | Face detection (live demo) |
| `matplotlib` / `seaborn` | Plots & visualizations |
| `Pillow` | Image I/O |

---
