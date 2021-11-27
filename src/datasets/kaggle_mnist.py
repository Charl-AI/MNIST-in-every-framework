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
    split. Test data is unlabelled."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        drop_last: bool,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last

        self.transform = None
        self.dims = (1, 28, 28)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            data = KaggleMNIST(
                data_dir=self.data_dir, train=True, transform=self.transform
            )
            n_val = int(len(data) * 0.2)
            n_train = len(data) - n_val
            self.train, self.val = random_split(data, [n_train, n_val])

        if stage == "test" or stage is None:
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

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--data_dir",
            help="directory containing test.csv and train.csv",
            type=str,
            default="data/kaggle_mnist",
        )
        parser.add_argument(
            "--batch_size",
            help="batch size",
            type=int,
            default=50,
        )
        parser.add_argument(
            "--num_workers",
            help="number of dataloader workers. 4*num_gpus is usually fine",
            type=int,
            default=4,
        )
        parser.add_argument(
            "--drop_last",
            help="whether to drop last batch from dataloader to keep batch sizes constant",
            type=bool,
            default=False,
        )
        return parent_parser
