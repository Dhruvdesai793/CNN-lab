import cv2
import torch
import matplotlib.pyplot as plt

from models.unet import UNet
from transforms import val_transform

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_CLASSES = 32
CHECKPOINT = 'best_model.pth'

def load_model():
    model = UNet(
        in_channels=3,
        num_classes=NUM_CLASSES,

    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location = DEVICE,
    )

    model.load_state_dict(
        checkpoint['model_state_dict']
    )

    model.eval()

    return model

def predict(model, image):
    transformed = val_transform(image = image)
    image = transformed['image']
    image = image.unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = model(image)

        prediction = torch.argmax(
            logits,
            dim = 1,
        )

    return prediction.squeeze(0).cpu().numpy()

def visualize(image, ground_truth, prediction):

    plt.figure(figsize = (18, 6))

    #original img
    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('input image')
    plt.axis("off")

    # Ground Truth
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB))
    plt.title("Ground Truth")
    plt.axis("off")

    # Prediction
    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap="tab20")
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():

    IMAGE_PATH = "data/CamVid/test/0001TP_008550.png"
    MASK_PATH = "data/CamVid/test_labels/0001TP_008550_L.png"

    image = cv2.imread(IMAGE_PATH)
    mask = cv2.imread(MASK_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {MASK_PATH}")

    model = load_model()

    prediction = predict(model, image)

    visualize(
        image=image,
        ground_truth=mask,
        prediction=prediction,
    )


if __name__ == "__main__":
    main()