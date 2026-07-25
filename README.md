<div align="center">

<img src="./assets/github-header-banner.png" width="100%">

<br>

# U-Net Semantic Segmentation

### Pixel-wise Scene Understanding using a Custom PyTorch Implementation

<p>
<img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white">
<img src="https://img.shields.io/badge/Albumentations-FFB000?style=for-the-badge">
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
</p>

*A complete semantic segmentation pipeline built from scratch using PyTorch.*

</div>

---

## Overview

This repository implements the original **U-Net** architecture from scratch in **PyTorch** for semantic segmentation on the **CamVid** dataset.

Rather than focusing only on the network, the project builds a complete deep learning pipeline including custom dataset handling, data augmentation, optimization, evaluation, checkpointing, visualization, and inference.

<p align="center">
<img src="./assets/u_net.webp" width="900">
</p>

---

# ✨ Features

- Original U-Net implementation in PyTorch
- Custom CamVid dataset pipeline
- Albumentations augmentation
- AdamW optimizer
- Cosine Annealing learning rate scheduler
- Gradient clipping
- Early stopping
- Automatic checkpoint saving
- Pixel Accuracy, Dice Score & Mean IoU evaluation
- Automatic training curve generation
- Standalone training, evaluation and inference scripts

---

# 🛠 Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| Framework | PyTorch |
| Dataset | CamVid |
| Image Processing | OpenCV |
| Augmentation | Albumentations |
| Visualization | Matplotlib |

---

# 📂 Project Structure

```text
.
├── assets/
├── models/
│   └── unet.py
├── results/
│   ├── loss_curve.png
│   ├── metrics_curve.png
│   └── metrics.txt
├── camvid_dataset.py
├── dataloader.py
├── transforms.py
├── metrics.py
├── utils.py
├── train.py
├── evaluate.py
├── infer.py
├── RESULTS.md
└── README.md
```

---

# 🚀 Quick Start

Clone the repository

```bash
git clone https://github.com/yourusername/unet-semantic-segmentation.git

cd unet-semantic-segmentation
```

Install dependencies

```bash
pip install -r requirements.txt
```

Train the model

```bash
python train.py
```

Evaluate the best checkpoint

```bash
python evaluate.py
```

Run inference

```bash
python infer.py
```

---

# 📊 Training Pipeline

```text
CamVid
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
CrossEntropyLoss
   │
   ▼
AdamW
   │
   ▼
CosineAnnealingLR
   │
   ▼
Validation
   │
   ├── Metrics
   ├── Checkpoints
   └── Training Curves
```

---

# 📈 Results

The repository includes:

- Loss curves
- Validation metrics
- Pixel Accuracy
- Dice Score
- Mean IoU
- Qualitative predictions

A complete experiment report is available in **[RESULTS.md](./RESULTS.md)**.

---

# 🗺 Possible Future Upgrades

- [x] Original U-Net
- [x] Modular training pipeline
- [x] Automatic evaluation
- [x] Inference pipeline
- [x] Training visualization
- [ ] Class-balanced loss
- [ ] Mixed precision training
- [ ] TensorBoard / Weights & Biases
- [ ] Attention U-Net

---

# 🙏 Acknowledgements

- U-Net: Convolutional Networks for Biomedical Image Segmentation
- CamVid Dataset
- PyTorch
- Albumentations

---

<div align="center">

**If you found this project useful, consider giving it a ⭐**

</div>