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


class KaggleMNISTDataModule(pl.LightningDataModule):
    """DataModule for MNIST downloaded from kaggle API.
    Splits the labelled data with an 80:20 train:val
    split. Test data is unlabelled.

    For using with the CLI, use:
    parser = KaggleMNISTDataModule.add_argparse_args(),
    then use:
    data = KaggleMNISTDataModule.from_argparse_args(args)

    Both methods come from the parent class, the add_argparse_args method
    uses the arguments from __init__ and the docstring to create the
    argument group.
    """

    def __init__(
        self,
        data_dir: str = "data/kaggle_mnist",
        batch_size: int = 50,
        num_workers: int = 4,
        drop_last: bool = False,
    ) -> None:
        """
        Args:
            data_dir: path to the data directory
            batch_size: batch size for dataloaders
            num_workers: number of workers for dataloaders, usually 4*num GPUs is fine,
                set to zero if you find a memory leak.
            drop_last: whether to drop the last batch to keep batch sizes constant
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last

        self.transform = None
        self.dims = (1, 28, 28)

    def setup(self, stage: Optional[str] = None):
        data = KaggleMNIST(data_dir=self.data_dir, train=True, transform=self.transform)
        n_val = int(len(data) * 0.2)
        n_train = len(data) - n_val
        self.train, self.val = random_split(data, [n_train, n_val])

        self.test = KaggleMNIST(
            data_dir=self.data_dir, train=False, transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )
