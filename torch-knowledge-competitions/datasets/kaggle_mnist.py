import os
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class KaggleMNIST(Dataset):
    """Dataset class for MNIST downloaded from Kaggle API"""

    def __init__(
        self,
        data_dir: str,
        train: bool,
        transform: Optional[callable] = None,
    ) -> None:
        """
        Args:
            data_dir: path to the data directory
            train: returns training dataset when true, testing set when False
            transform: optional transforms to apply. Not advised for MNIST.
        """

        self.data_dir = data_dir
        self.transform = transform
        self.train = train

        if self.train:
            df = pd.read_csv(os.path.join(data_dir, "train.csv"))
            self.labels = torch.Tensor(df["label"].values)
            self.imgs = df.drop(labels="label", axis=1)
            self.imgs = torch.Tensor(self.imgs.values)

        else:
            df = pd.read_csv(os.path.join(data_dir, "test.csv"))
            self.imgs = torch.Tensor(df.values)

        self.imgs = self.imgs.unflatten(dim=1, sizes=(28, 28))
        self.imgs = self.imgs.unsqueeze(1)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index: int) -> tuple:

        img = self.imgs[index]

        if self.transform:
            img = self.transform(img)

        if self.train:
            label = self.labels[index]
            return img, label
        return img
