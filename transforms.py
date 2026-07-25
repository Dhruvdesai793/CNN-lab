import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_SIZE = 256

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),

    A.HorizontalFlip(p=0.5),

    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.10,
        rotate_limit=10,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.5,
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5,
    ),

    A.GaussNoise(
        std_range=(0.02, 0.08),
        p=0.2,
    ),

    A.Normalize(
        mean=MEAN,
        std=STD,
    ),

    ToTensorV2(),
])


val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),

    A.Normalize(
        mean=MEAN,
        std=STD,
    ),

    ToTensorV2(),
])