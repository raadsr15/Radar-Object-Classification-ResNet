# Radar Object Classification with ResNet50

This repository presents a complete workflow for object classification using radar range-Doppler spectrograms and deep learning. The goal is to distinguish between cars, drones, and people using frequency-modulated continuous wave (FMCW) radar data, which is particularly useful in surveillance, security, and smart mobility applications where vision-based sensors may fail.

Starting from public radar datasets in CSV format, the pipeline includes automatic conversion of radar matrices into 2D spectrogram images for machine learning. Data is thoroughly organized and visualized to ensure quality and interpretability. All images are flattened into class-specific folders and then split into train, validation, and test sets for robust model evaluation.

A ResNet50 convolutional neural network is trained from scratch on these spectrogram images, leveraging data augmentation and normalization for best results. The network achieves strong classification performance, as demonstrated by high test accuracy and detailed per-class metrics (precision, recall, F1-score). Training progress, validation curves, and a confusion matrix are visualized for transparent model assessment. The repository also includes scripts for data processing, training, evaluation, and visualization, making it easy to reproduce results or extend to new classes.

# Dataset Description
This project utilizes the Real Doppler RAD-DAR Database, a large public dataset designed for radar-based object classification research.

The RAD-DAR database was collected using a state-of-the-art frequency-modulated continuous wave (FMCW) radar system (Digital Array Receiver, 8.75 GHz center frequency, 500 MHz bandwidth) developed by the UPM Microwave and Radar Group. The dataset contains over 17,000 samples spanning three object categories: Cars, Drones, and People.

Each sample consists of a range-Doppler matrix—a 2D radar “image” that encodes both distance and velocity information—provided in CSV format. Samples are organized by object class and subdirectory, with each file containing a processed radar measurement for a single detection event. All data is manually labeled to ensure ground truth quality.

###  Dataset Access

The RAD-DAR Database is freely available from Kaggle:

🔗 [Real Doppler RAD-DAR Database – Kaggle](https://www.kaggle.com/datasets/iroldan/real-doppler-raddar-database)

The dataset was collected and released by the UPM Microwave and Radar Group and is detailed in the associated documentation on the Kaggle page.

---

###  Data Characteristics

Each sample in the dataset is a range-Doppler matrix (radar “image”) representing a single detection event, saved as a CSV file.
The data was collected with a frequency-modulated continuous wave (FMCW) radar (8.75 GHz, 500 MHz bandwidth) and includes three classes:

**Cars**
**Drones**
**People**

Total samples: Over 17,000
Format: CSV, each encoding the radar matrix for one scene

---

###  File Format Overview

| Item    | Description                           |
|---------|---------------------------------------|
| Rows    | Range (distance) bins                 |
| Columns | Doppler (velocity) bins               |
| Values  | Radar signal intensity (amplitude)    |
| File    | `.csv` (one sample per file)          |

All samples are organized by class and subfolder according to the experimental setup.


---

###  Data Split Strategy

To ensure model robustness and prevent data leakage:

- **Training & Validation Sets:**  
  Randomly selected 70% (train) and 15% (validation) of samples per class

- **Test Set:**  
  Remaining 15% of samples per class

All splits are stratified to preserve class balance and diversity.

---
## Pipeline Overview

### 1. **CSV to PNG Conversion**
- Radar matrices are normalized and converted to PNG images (`matplotlib.pyplot.imsave`) for easy deep learning ingestion.

### 2. **Data Visualization**
- Random spectrograms are visualized to verify conversion and interpret class differences.
- ![image](https://github.com/user-attachments/assets/77114e49-2684-42fa-a945-7ee506fea801)


### 3. **Flattening Class Folders**
- All images are copied into a single folder per class (`flat_dataset/`), removing any deep subfolder hierarchy for fast access.

### 4. **Model Training**
- A ResNet50 model is trained on the spectrograms (resized to 64x64).
- Data augmentation and normalization are applied.
- Training and validation curves are logged and plotted.

### 5. **Evaluation**
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
