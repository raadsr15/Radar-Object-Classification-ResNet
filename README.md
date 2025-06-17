# Radar Object Classification with ResNet50

This repository presents a complete workflow for object classification using radar range-Doppler spectrograms and deep learning. The goal is to distinguish between cars, drones, and people using frequency-modulated continuous wave (FMCW) radar data, which is particularly useful in surveillance, security, and smart mobility applications where vision-based sensors may fail.

Starting from public radar datasets in CSV format, the pipeline includes automatic conversion of radar matrices into 2D spectrogram images for machine learning. Data is thoroughly organized and visualized to ensure quality and interpretability. All images are flattened into class-specific folders and then split into train, validation, and test sets for robust model evaluation.

A ResNet50 convolutional neural network is trained from scratch on these spectrogram images, leveraging data augmentation and normalization for best results. The network achieves strong classification performance, as demonstrated by high test accuracy and detailed per-class metrics (precision, recall, F1-score). Training progress, validation curves, and a confusion matrix are visualized for transparent model assessment. The repository also includes scripts for data processing, training, evaluation, and visualization, making it easy to reproduce results or extend to new classes.

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
