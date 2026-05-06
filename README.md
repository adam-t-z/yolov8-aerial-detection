# Object Detection: Car Detection on Aerial Images

A modular, production-ready pipeline for training YOLOv8n object detection models on custom datasets with automated preprocessing, training, validation, and interactive deployment.
---

![Demo](media/demo.gif)

---

## 🌐 Try the Deployed Model

You can try the deployed model online using the following link:

[Deployed Model on Streamlit](https://yolov8-ariel-detection.streamlit.app/)

---

## 📋 Project Overview

This project implements a complete machine learning pipeline for detecting cars in aerial imagery using YOLOv8 (nano). The system converts XML annotations to YOLO format, trains a model, validates performance, and provides an interactive web interface for inference.

**Dataset:** 230 aerial images | 1 class<br>
**Model:** YOLOv8n (nano - lightweight)  
**Framework:** Ultralytics YOLO, Streamlit  
**Python:** 3.11+


## Note on Dataset

Before running the preprocessing step, ensure that the `data-original` directory contains the full dataset. The current directory includes only 10 images to save space.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
# Install uv if not already installed:
pip install uv

# Install project dependencies:
uv sync
```

### 2. Run the Complete Pipeline
```bash
# Preprocess XML annotations to YOLO format
uv run pipeline/1-preprocess.py

# Train YOLOv8n model
uv run pipeline/2-train.py

# Validate on test set
uv run pipeline/3-validate.py

# Launch interactive app
uv run streamlit run app.py
```

  > `note`: specify parameters in `config.yaml` file

---

## 📁 Project Structure

```
.
├── README.md                 ← This file
├── config.yaml              ← All configuration (hyperparameters, paths, etc.)
├── pyproject.toml           ← Project metadata and dependencies
├── app.py                   ← Streamlit inference web app
│
├── pipeline/                ← Modular training pipeline
│   ├── 1-preprocess.py      ← Convert XML → YOLO format
│   ├── 2-train.py           ← Train YOLOv8n model
│   └── 3-validate.py        ← Evaluate model on test set
│
├── data/                    ← Preprocessed YOLO-format data
│   ├── data.yaml            ← Dataset config for training
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── data-original/           ← Original XML annotations (source)
│   └── *.xml
│
├── best.pt              ← Added after training for deployment
│
└── runs/                   ← Training outputs (auto-created)
    └── detect/experiment_*/
        ├── weights/
        │   ├── best.pt     ← Best trained model
        │   └── last.pt
        ├── results.csv
        └── ...
```

---

## Inference Locally on Images and Video
Interactive Streamlit interface for real-time detection:
- Upload images or videos
- Adjust confidence threshold
- View annotated results
- Auto-loads best trained model or fallback to `yolov8n.pt`

```bash
uv run streamlit run app.py
```

---

## ⚙️ Configuration

All settings are centralized in **`config.yaml`**—no hardcoded values:

```yaml
# Training hyperparameters
training:
  epochs: 100              # Number of training epochs
  batch_size: 8            # Batch size
  learning_rate: 0.001     # Learning rate
  patience: 20             # Early stopping patience
  device: "cpu"            # "cpu" for CPU | "0" for GPU
  augment: true            # Enable data augmentation

# Validation settings
validation:
  confidence_threshold: 0.5    # Detection confidence cutoff
  iou_threshold: 0.45          # IoU threshold for NMS

# Preprocessing
preprocess:
  seed: 42                 # Random seed for reproducibility
  split: [0.7, 0.2, 0.1]  # Train/val/test split ratios
```

To customize: Edit `config.yaml` and rerun relevant pipeline stages.

---

## 📄 Project Report

For a detailed overview of the project, refer to the [Project Report](REPORT.MD).

---

## 🔧 Development

### Code Quality
```bash
# Lint and code formatting with ruff
uv run ruff check .
```

### Install Development Dependencies
```bash
uv sync --all-extras  # Includes pytest, ruff
```

---

## 🛠️ Common Tasks

**Use GPU for training:**
```yaml
# config.yaml
training:
  device: "0"  # "cpu" for CPU | "0" for GPU
```

**Increase dataset size:** Add more images/XML files to `data-original/`, then rerun:
```bash
uv run pipeline/1-preprocess.py
```

**Adjust training duration:** Modify in `config.yaml`:
```yaml
training:
  epochs: 200        # More epochs
  patience: 30       # Later early stopping
```

---

## 📚 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [YOLO Data Format](https://docs.ultralytics.com/datasets/detect/)

