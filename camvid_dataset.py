from pathlib import Path
import csv

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class CamVidDataset(Dataset):
    def __init__(self, root, split='train', transform= None):
        self.root = Path(root)
        self.transform = transform

        self.image_dir = self.root / split
        self.mask_dir = self.root / f"{split}_labels"

        self.images = sorted(self.image_dir.glob("*.png"))

        self.color_to_class = self._load_class_dict()

    def __len__(self):
        return len(self.images)

    def _load_class_dict(self):
        color_to_class = {}

        with open(self.root / "class_dict.csv", newline="") as f:
            reader = csv.DictReader(f)

            for idx, row in enumerate(reader):
                color = (
                    int(row['r']),
                    int(row['g']),
                    int(row['b'])
                )

                color_to_class[color] = idx

        return color_to_class

    def __getitem__(self, index):
        image_path = self.images[index]

        mask_name = image_path.stem + "_L.png"
        mask_path = self.mask_dir / mask_name


        # imread() doesn't take in path object, so convert it to str
        #read img
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #read mask
        mask = cv2.imread(str(mask_path))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # convert RGB masks to class-index mask

        mask_indices = np.zeros(mask.shape[:2], dtype = np.uint8)

        for color, class_id in self.color_to_class.items():
            matches = np.all(mask == color, axis = -1)
            mask_indices[matches] = class_id

        # apply augmentations

        if self.transform:
            augmented = self.transform(image = image, mask = mask_indices)
            image = augmented["image"]
            mask_indices = augmented["mask"].long() #cross entropy expects .long()


        else:
            image = torch.from_numpy(image).permute(2,0,1).float() / 255.0
            mask_indices = torch.from_numpy(mask_indices).long()

        return image, mask_indices