from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_history(
    train_losses,
    val_losses,
    pixel_accs,
    dice_scores,
    mious,
    save_dir="results",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    # Loss
    plt.figure(figsize=(8, 5))

    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_dir / "loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Metrics
    plt.figure(figsize=(8, 5))

    plt.plot(epochs, pixel_accs, label="Pixel Accuracy")
    plt.plot(epochs, dice_scores, label="Dice")
    plt.plot(epochs, mious, label="mIoU")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_dir / "metrics_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved training plots.")