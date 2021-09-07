import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional
import pandas as pd
import torch


class MNISTKaggle(Dataset):
    """Dataset class for MNIST downloaded from Kaggle API"""

    def __init__(
        self,
        data_dir: str = r"data/mnist_kaggle",
        train: bool = True,
        transform: Optional[callable] = None,
    ) -> None:
        """
        Args:
            data_dir: path to the data directory
            train: returns training dataset when true, testing set when False
            transform: optional transforms to apply to training data. Not advised for MNIST.
        """

        self.data_dir = data_dir
        self.transform = transform
        self.train = train

        if self.train:
            df = pd.read_csv(data_dir + "/train.csv")
            self.labels = torch.Tensor(df["label"].values).long()
            self.imgs = df.drop(labels="label", axis=1)
            self.imgs = torch.Tensor(self.imgs.values)

        else:
            df = pd.read_csv(data_dir + "/test.csv")
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


class MNISTKaggleModule(pl.LightningDataModule):
    """DataModule for MNIST downloaded from kaggle API."""

    def __init__(
        self, data_dir: str = r"data/mnist_kaggle", batch_size: int = 32
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = None

        self.dims = (1, 28, 28)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            data = MNISTKaggle(
                data_dir=self.data_dir, train=True, transform=self.transform
            )
            n_val = int(len(data) * 0.2)
            n_train = len(data) - n_val
            self.train, self.val = random_split(data, [n_train, n_val])

        if stage == "test" or stage is None:
            self.test = MNISTKaggle(
                data_dir=self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
