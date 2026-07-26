
import os
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from models.unet import UNet
from transforms import val_transform
from utils import load_checkpoint


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 32

CHECKPOINT = "best_model.pth"
IMAGE_PATH = "data/CamVid/test/0001TP_008550.png"
MASK_PATH = "data/CamVid/test_labels/0001TP_008550_L.png"

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# CamVid palette (first 32 classes)
CAMVID_COLORS = np.array([
    [128,128,128],[128,0,0],[192,192,128],[128,64,128],
    [0,0,192],[128,128,0],[192,128,128],[64,64,128],
    [64,0,128],[64,64,0],[0,128,192],[0,0,0],
    [64,128,128],[128,0,192],[192,0,64],[128,128,192],
    [64,0,192],[192,128,64],[128,64,64],[64,192,128],
    [64,64,64],[192,0,128],[64,128,64],[128,192,192],
    [0,0,64],[192,64,128],[128,128,64],[192,0,192],
    [128,64,192],[64,128,192],[0,128,64],[192,192,0],
], dtype=np.uint8)

CLASS_NAMES = [
    "Animal","Archway","Bicyclist","Bridge","Building","Car","CartLuggagePram",
    "Child","ColumnPole","Fence","LaneMkgsDriv","LaneMkgsNonDriv","MiscText",
    "MotorcycleScooter","OtherMoving","ParkingBlock","Pedestrian","Road","RoadShoulder",
    "Sidewalk","SignSymbol","Sky","SUVPickupTruck","TrafficCone","TrafficLight",
    "Train","Tree","TruckBus","Tunnel","VegetationMisc","Void","Wall"
]


def load_model():
    model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
    load_checkpoint(model=model, optimizer=None, path=CHECKPOINT, device=DEVICE)
    model.eval()
    return model


def decode_mask(mask):
    mask = np.clip(mask, 0, len(CAMVID_COLORS)-1)
    return CAMVID_COLORS[mask]


@torch.no_grad()
def predict(model, image):
    original_h, original_w = image.shape[:2]
    tensor = val_transform(image=image)["image"].unsqueeze(0).to(DEVICE)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    logits = model(tensor)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    pred = logits.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
    pred = cv2.resize(pred, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    return pred, elapsed


def save_metrics(image, elapsed_ms, pred):
    fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0
    classes = np.unique(pred)

    with open(RESULTS_DIR / "metrics.txt", "w") as f:
        f.write("Model: UNet\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Input Size: {image.shape[1]}x{image.shape[0]}\n")
        f.write(f"Inference Time: {elapsed_ms:.2f} ms\n")
        f.write(f"FPS: {fps:.2f}\n\n")
        f.write("Detected Classes:\n")
        for c in classes:
            if c < len(CLASS_NAMES):
                f.write(f"- {CLASS_NAMES[c]}\n")


def visualize(image, gt, pred, elapsed_ms):
    pred_rgb = decode_mask(pred)

    overlay = cv2.addWeighted(image, 0.6, pred_rgb, 0.4, 0)

    cv2.imwrite(str(RESULTS_DIR / "prediction.png"),
                cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(RESULTS_DIR / "overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    fig, ax = plt.subplots(2, 2, figsize=(16, 10))

    ax[0,0].imshow(image)
    ax[0,0].set_title("Original")

    ax[0,1].imshow(gt)
    ax[0,1].set_title("Ground Truth")

    ax[1,0].imshow(pred_rgb)
    ax[1,0].set_title(f"Prediction ({elapsed_ms:.2f} ms | {1000/elapsed_ms:.1f} FPS)")

    ax[1,1].imshow(overlay)
    ax[1,1].set_title("Overlay")

    for a in ax.ravel():
        a.axis("off")

    patches = [
        Patch(facecolor=np.array(CAMVID_COLORS[i])/255.0,
              edgecolor="black",
              label=CLASS_NAMES[i])
        for i in np.unique(pred) if i < len(CLASS_NAMES)
    ]
    if patches:
        fig.legend(handles=patches, loc="center right", fontsize=8)

    plt.tight_layout(rect=[0,0,0.88,1])
    plt.savefig(RESULTS_DIR / "comparison.png", dpi=200)
    plt.show()


def main():
    image = cv2.cvtColor(cv2.imread(IMAGE_PATH), cv2.COLOR_BGR2RGB)
    gt = cv2.cvtColor(cv2.imread(MASK_PATH), cv2.COLOR_BGR2RGB)

    model = load_model()
    pred, elapsed = predict(model, image)

    fps = 1000.0 / elapsed

    print("=" * 50)
    print(f"Device          : {DEVICE}")
    print(f"Input Size      : {image.shape[1]} x {image.shape[0]}")
    print(f"Inference Time  : {elapsed:.2f} ms")
    print(f"FPS             : {fps:.2f}")
    print(f"Classes Found   : {len(np.unique(pred))}")
    print("\nDetected Classes")
    print("-" * 20)
    for idx in np.unique(pred):
        if idx < len(CLASS_NAMES):
            print(CLASS_NAMES[idx])

    save_metrics(image, elapsed, pred)
    visualize(image, gt, pred, elapsed)


if __name__ == "__main__":
    main()
