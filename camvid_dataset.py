from pathlib import Path
import csv

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class CamVidDataset(Dataset):

    def __init__(
        self,
        root,
        split="train",
        transform=None,
    ):
        self.root = Path(root)
        self.transform = transform

        self.image_dir = self.root / split
        self.mask_dir = self.root / f"{split}_labels"

        self.images = sorted(self.image_dir.glob("*.png"))

        self.color_to_class = self.load_class_dict()

    def __len__(self):
        return len(self.images)

    def load_class_dict(self):

        mapping = {}

        with open(self.root / "class_dict.csv") as f:

            reader = csv.DictReader(f)

            for idx, row in enumerate(reader):

                mapping[
                    (
                        int(row["r"]),
                        int(row["g"]),
                        int(row["b"]),
                    )
                ] = idx

        return mapping

    def __getitem__(self, index):

        image_path = self.images[index]

        mask_path = self.mask_dir / f"{image_path.stem}_L.png"

        image = cv2.cvtColor(
            cv2.imread(str(image_path)),
            cv2.COLOR_BGR2RGB,
        )

        mask = cv2.cvtColor(
            cv2.imread(str(mask_path)),
            cv2.COLOR_BGR2RGB,
        )

        mask_indices = np.zeros(
            mask.shape[:2],
            dtype=np.uint8,
        )

        for color, cls in self.color_to_class.items():
            mask_indices[np.all(mask == color, axis=-1)] = cls

        if self.transform:

            transformed = self.transform(
                image=image,
                mask=mask_indices,
            )

            image = transformed["image"]
            mask_indices = transformed["mask"].long()

        else:

            image = (
                torch.from_numpy(image)
                .permute(2, 0, 1)
                .float()
                / 255.0
            )

            mask_indices = torch.from_numpy(mask_indices).long()

        return image, mask_indices