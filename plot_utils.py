from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_history(
    history,
    save_dir="results",
):
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        save_dir / "loss_curve.png",
        dpi=300,
    )

    plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(epochs, history["pixel_acc"], label="Pixel Accuracy")
    plt.plot(epochs, history["dice"], label="Dice")
    plt.plot(epochs, history["miou"], label="mIoU")

    plt.xlabel("Epoch")
    plt.ylabel("Score")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        save_dir / "metrics_curve.png",
        dpi=300,
    )

    plt.close()