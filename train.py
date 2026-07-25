import torch
import torch.nn as nn
from tqdm import tqdm

from dataloader import get_dataloaders
from models.unet import UNet
from utils import save_checkpoints
from metrics import pixel_accuracy, dice_score, mean_iou
from plot_utils import plot_training_history


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 32
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
EPOCHS = 100


def main():

    train_loader, val_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
    )

    model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")

    # History
    train_losses = []
    val_losses = []

    pixel_accs = []
    dice_scores = []
    mious = []

    # Training
    for epoch in range(EPOCHS):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_loss, pixel_acc, dice, miou = validate(
            model,
            val_loader,
            criterion,
        )

        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        pixel_accs.append(pixel_acc)
        dice_scores.append(dice)
        mious.append(miou)

        print(
            f"Epoch {epoch+1}/{EPOCHS}"
            f" | Train Loss: {train_loss:.4f}"
            f" | Val Loss: {val_loss:.4f}"
            f" | Pixel Acc: {pixel_acc:.4f}"
            f" | Dice: {dice:.4f}"
            f" | mIoU: {miou:.4f}"
        )

        if val_loss < best_loss:

            best_loss = val_loss

            save_checkpoints(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                filename="best_model.pth",
            )

    # Plot Training Curves
    plot_training_history(
        train_losses=train_losses,
        val_losses=val_losses,
        pixel_accs=pixel_accs,
        dice_scores=dice_scores,
        mious=mious,
    )


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
):

    model.train()

    running_loss = 0.0

    progress_bar = tqdm(train_loader)

    for images, masks in progress_bar:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(train_loader)


def validate(
    model,
    val_loader,
    criterion,
):

    model.eval()

    running_loss = 0.0
    pixel_acc = 0.0
    dice = 0.0
    miou = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, masks)

            running_loss += loss.item()

            pixel_acc += pixel_accuracy(outputs, masks)
            dice += dice_score(outputs, masks, NUM_CLASSES)
            miou += mean_iou(outputs, masks, NUM_CLASSES)

    num_batches = len(val_loader)

    return (
        running_loss / num_batches,
        pixel_acc / num_batches,
        dice / num_batches,
        miou / num_batches,
    )


if __name__ == "__main__":
    main()