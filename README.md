<div align="center">

<img src="./assets/github-header-banner.png" alt="GitHub Header Banner" width="100%"/>

<br>

# U-Net Semantic Segmentation

### Pixel-wise Scene Understanding with a Custom U-Net Implementation in PyTorch

<p>
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/Albumentations-FFB000?style=for-the-badge">
</p>

*A complete semantic segmentation pipeline built from scratch using PyTorch.*

</div>

---

## Overview

Semantic segmentation is the task of assigning a semantic label to **every pixel** in an image. Unlike image classification or object detection, semantic segmentation provides dense scene understanding, making it a fundamental technique for applications such as autonomous driving, robotics, medical imaging, and satellite imagery.

This project implements the **original U-Net architecture** entirely from scratch using **PyTorch**. Beyond the model itself, the repository contains a complete end-to-end training pipeline including custom dataset handling, data augmentation, evaluation metrics, checkpointing, inference, and automatic visualization of training progress.

The primary objective of this project was not only to reproduce the U-Net architecture, but also to understand how a modern deep learning vision project is engineered from data loading to deployment-ready inference.

---

# Repository Highlights

- 🚀 Complete semantic segmentation pipeline built from scratch
- 🧠 Original U-Net architecture implemented in PyTorch
- 📦 Modular and reusable project structure
- 🎨 Albumentations-based augmentation pipeline
- 📊 Automatic training curve generation
- 📈 Multiple evaluation metrics (Pixel Accuracy, Dice Score, Mean IoU)
- 💾 Automatic checkpoint saving
- 🔍 Separate scripts for training, evaluation, and inference

---

# Table of Contents

- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Training Pipeline](#training-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Results](#results)
- [What I Learned](#what-i-learned)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)

---

# Model Architecture

<p align="center">
    <img src="./assets/u_net.webp" width="900" alt="U-Net Architecture">
</p>

The network follows the original **encoder-decoder U-Net architecture**. The encoder progressively extracts increasingly abstract semantic features through repeated convolution and downsampling, while the decoder gradually reconstructs the spatial resolution using transposed convolutions.

Skip connections bridge corresponding encoder and decoder stages, allowing the model to preserve fine-grained localization information that would otherwise be lost during downsampling. This design enables accurate pixel-level predictions while maintaining strong semantic understanding.

---

# Dataset

The model is trained using the **CamVid (Cambridge-driving Labeled Video Database)** dataset.

CamVid is one of the earliest semantic segmentation benchmarks and contains densely annotated urban driving scenes captured from a vehicle-mounted camera.

### Dataset Characteristics

- Urban street scenes
- Pixel-level annotations
- RGB images
- 32 semantic classes
- Road, buildings, sky, vehicles, pedestrians, trees, sidewalks, traffic signs, poles, and more

---

# Project Structure

```text
.
├── assets/
│   ├── github-header-banner.png
│   └── u_net.webp
│
├── models/
│   └── unet.py
│
├── results/
│   ├── loss_curve.png
│   ├── metrics_curve.png
│   └── ...
│
├── camvid_dataset.py
├── dataloader.py
├── transforms.py
├── metrics.py
├── plot_utils.py
├── utils.py
├── train.py
├── evaluate.py
├── infer.py
├── RESULTS.md
├── requirements.txt
└── README.md
```

---

# Training Pipeline

```text
CamVid Dataset
        │
        ▼
Albumentations
        │
        ▼
DataLoader
        │
        ▼
U-Net
        │
        ▼
CrossEntropy Loss
        │
        ▼
Adam Optimizer
        │
        ▼
Validation
        │
        ▼
Evaluation Metrics
        │
        ▼
Checkpoint Saving
        │
        ▼
Training Curves
```

---

# Evaluation Metrics

The model is evaluated using several complementary metrics to provide a comprehensive assessment of segmentation quality.

| Metric | Description |
|---------|-------------|
| **Validation Loss** | Cross-entropy loss on the validation set |
| **Pixel Accuracy** | Percentage of correctly classified pixels |
| **Dice Score** | Measures overlap between prediction and ground truth |
| **Mean IoU** | Average Intersection over Union across all classes |

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/unet-semantic-segmentation.git
cd unet-semantic-segmentation
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Training

```bash
python train.py
```

Training automatically performs:

- Model optimization
- Validation after every epoch
- Best checkpoint saving
- Training history recording
- Automatic curve generation

---

# Evaluation

```bash
python evaluate.py
```

The evaluation script computes:

- Validation Loss
- Pixel Accuracy
- Dice Score
- Mean IoU

---

# Inference

```bash
python infer.py
```

Runs inference on validation images and visualizes:

- Input Image
- Ground Truth Mask
- Predicted Segmentation Mask

---

# Results

Training curves, quantitative metrics, qualitative predictions, and experiment observations are documented in **[RESULTS.md](./RESULTS.md)**.

---

# What I Learned

Developing this project provided hands-on experience with the complete semantic segmentation workflow, including:

- Building the original U-Net architecture from scratch
- Designing encoder-decoder neural networks
- Understanding skip connections
- Creating custom PyTorch datasets
- Data augmentation with Albumentations
- Pixel-wise loss functions
- Semantic segmentation evaluation metrics
- Model checkpointing
- Building reusable training pipelines
- Modular deep learning project organization

---

# Future Improvements

Some potential future extensions include:

- Attention U-Net
- U-Net++
- DeepLabV3+
- Mixed Precision Training
- Learning Rate Scheduling
- Early Stopping
- Hyperparameter Optimization
- TensorBoard / Weights & Biases integration
- Multi-dataset benchmarking

---

# Acknowledgements

This project was inspired by and built upon the following resources:

- **U-Net: Convolutional Networks for Biomedical Image Segmentation**
- **CamVid Dataset**
- **PyTorch**
- **Albumentations**

---

<div align="center">

### ⭐ If you found this project useful, consider starring the repository!

</div>