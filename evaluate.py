from pathlib import Path

import torch
import torch.nn as nn

from dataloader import get_dataloaders
from models.unet import UNet
from train import validate
from utils import load_checkpoint


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 32
BATCH_SIZE = 8

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def main():

    _, val_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
    )

    model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    load_checkpoint(
        model=model,
        optimizer=None,
        path="best_model.pth",
        device=DEVICE,
    )

    val_loss, pixel, dice, miou = validate(
        model,
        val_loader,
        criterion,
    )

    report = f"""
================ Evaluation ================

Model            : U-Net
Dataset          : CamVid

Validation Loss  : {val_loss:.4f}
Pixel Accuracy   : {pixel:.4f}
Dice Score       : {dice:.4f}
Mean IoU         : {miou:.4f}

============================================
"""

    print(report)

    with open(RESULTS_DIR / "metrics.txt", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()