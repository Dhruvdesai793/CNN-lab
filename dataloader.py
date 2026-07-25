from torch.utils.data import DataLoader

from camvid_dataset import CamVidDataset
from transforms import train_transform, val_transform


def get_dataloaders(
    batch_size=8,
    num_workers=4,
):

    train_dataset = CamVidDataset(
        root="data/CamVid",
        split="train",
        transform=train_transform,
    )

    val_dataset = CamVidDataset(
        root="data/CamVid",
        split="val",
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader