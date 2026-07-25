import torch
import torch.nn as nn
from tqdm import tqdm

from dataloader import get_dataloaders
from metrics import pixel_accuracy, dice_score, mean_iou
from models.unet import UNet
from plot_utils import plot_training_history
from utils import save_checkpoint


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 32
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING = 15


def train_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0.0

    progress = tqdm(loader, leave=False)

    for images, masks in progress:

        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        total_loss += loss.item()

        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion):

    model.eval()

    loss = 0.0
    pixel = 0.0
    dice = 0.0
    miou = 0.0

    for images, masks in loader:

        images = images.to(DEVICE, non_blocking=True)
        masks = masks.to(DEVICE, non_blocking=True)

        outputs = model(images)

        loss += criterion(outputs, masks).item()
        pixel += pixel_accuracy(outputs, masks)
        dice += dice_score(outputs, masks, NUM_CLASSES)
        miou += mean_iou(outputs, masks, NUM_CLASSES)

    n = len(loader)

    return (
        loss / n,
        pixel / n,
        dice / n,
        miou / n,
    )


def main():

    train_loader, val_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
    )

    model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "val_loss": [],
        "pixel_acc": [],
        "dice": [],
        "miou": [],
    }

    best_loss = float("inf")
    patience = 0

    for epoch in range(EPOCHS):

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_loss, pixel, dice, miou = validate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["pixel_acc"].append(pixel)
        history["dice"].append(dice)
        history["miou"].append(miou)

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:03}/{EPOCHS}"
            f" | LR {lr:.2e}"
            f" | Train {train_loss:.4f}"
            f" | Val {val_loss:.4f}"
            f" | Pixel {pixel:.4f}"
            f" | Dice {dice:.4f}"
            f" | mIoU {miou:.4f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss
            patience = 0

            save_checkpoint(
                model,
                optimizer,
                epoch,
                "best_model.pth",
            )

        else:

            patience += 1

            if patience >= EARLY_STOPPING:
                print("\nEarly stopping.")
                break

    plot_training_history(history)


if __name__ == "__main__":
    main()