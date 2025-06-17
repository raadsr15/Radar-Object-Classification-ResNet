# Radar Object Classification with ResNet50

This repository implements an end-to-end pipeline for classifying objects (Cars, Drones, and People) from FMCW radar range-Doppler spectrograms using deep learning (ResNet50). The workflow spans raw radar matrix conversion, visualization, dataset flattening and splitting, model training, and evaluation.

## Dataset

This project uses a **public FMCW radar dataset** (e.g., RAD-DAR or similar) containing thousands of labeled radar returns in CSV format, organized by object class (`Cars`, `Drones`, `People`). Each radar frame is a 2D range-Doppler matrix, saved per sample.

- Raw data: `/dataset/[class]/[subfolder]/xxx.csv`
- PNGs for ML: `/images/[class]/[subfolder]/xxx.png`

**(Refer to the [RAD-DAR Database on Kaggle](https://www.kaggle.com/datasets/ignaciojeria/real-doppler-rad-dar-database) for more details.)**

---

## Pipeline Overview

### 1. **CSV to PNG Conversion**
- Radar matrices are normalized and converted to PNG images (`matplotlib.pyplot.imsave`) for easy deep learning ingestion.

### 2. **Data Visualization**
- Random spectrograms are visualized to verify conversion and interpret class differences.

### 3. **Flattening Class Folders**
- All images are copied into a single folder per class (`flat_dataset/`), removing any deep subfolder hierarchy for fast access.

### 4. **Dataset Split**
- Images are randomly and reproducibly split into `train` (70%), `val` (15%), and `test` (15%) for each class.

### 5. **Model Training**
- A ResNet50 model is trained on the spectrograms (resized to 64x64).
- Data augmentation and normalization are applied.
- Training and validation curves are logged and plotted.

### 6. **Evaluation**
- Final model is tested on the holdout test set.
- Outputs: test accuracy, per-class metrics, confusion matrix, and visualizations of model predictions.

---

## Results

| Metric         | Cars  | Drones | People | Avg.   |
|----------------|-------|--------|--------|--------|
| Precision      | 0.923 | 0.901  | 0.996  | 0.940  |
| Recall         | 0.913 | 0.923  | 0.988  | 0.941  |
| F1 Score       | 0.918 | 0.912  | 0.992  | 0.945  |
| Test Accuracy  |         **0.944**                 |

- **Confusion Matrix** and prediction visualizations are included in the notebook/scripts.

---

## Installation and Usage

1. **Clone the repository:**
    ```bash
    git clone https://github.com/raadsr15/Radar-Object-Classification-ResNet.git
    cd Radar-Object-Classification-ResNet
