from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def make_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )


class RetinalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int, train: bool):
        self.df = df.reset_index(drop=True)
        self.tfms = make_transforms(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        with Image.open(row["image_path"]).convert("RGB") as img:
            x = self.tfms(img)
        y = torch.tensor(int(row["class_id"]), dtype=torch.long)
        return x, y

