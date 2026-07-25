import cv2
import matplotlib.pyplot as plt
import torch

from models.unet import UNet
from transforms import val_transform
from utils import load_checkpoint


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES = 32

IMAGE_PATH = "data/CamVid/test/0001TP_008550.png"
MASK_PATH = "data/CamVid/test_labels/0001TP_008550_L.png"


def load_model():

    model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
    ).to(DEVICE)

    load_checkpoint(
        model=model,
        optimizer=None,
        path="best_model.pth",
        device=DEVICE,
    )

    model.eval()

    return model


@torch.no_grad()
def predict(model, image):

    image = val_transform(image=image)["image"]

    image = image.unsqueeze(0).to(DEVICE)

    prediction = model(image).argmax(dim=1)

    return prediction.squeeze().cpu().numpy()


def visualize(image, mask, prediction):

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap="tab20")
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():

    image = cv2.cvtColor(
        cv2.imread(IMAGE_PATH),
        cv2.COLOR_BGR2RGB,
    )

    mask = cv2.cvtColor(
        cv2.imread(MASK_PATH),
        cv2.COLOR_BGR2RGB,
    )

    model = load_model()

    prediction = predict(
        model,
        image,
    )

    visualize(
        image,
        mask,
        prediction,
    )


if __name__ == "__main__":
    main()