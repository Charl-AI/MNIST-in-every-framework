import os
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


class KaggleTweets(Dataset):
    """Dataset class for disaster tweets downloaded through Kaggle API."""

    def __init__(
        self,
        data_dir: str,
        train: bool,
    ) -> None:
        """
        Args:
            data_dir: path to the data directory
            train: returns training dataset when true, testing set when False
        """
        self.data_dir = data_dir
        self.train = train
        df = pd.read_csv(
            os.path.join(data_dir, f"{'train' if self.train else 'test'}.csv")
        )

        if self.train:
            self.targets = df["target"]

        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            for text in df["text"]
        ]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        if self.train:
            return self.texts[index], self.targets[index]
        else:
            return self.texts[index]


class KaggleTweetsDataModule(pl.LightningDataModule):
    """DataModule for MNIST downloaded from kaggle API.
    Splits the labelled data with an 80:20 train:val
    split. Test data is unlabelled.
    """

    def __init__(
        self,
        data_dir: str = "data/kaggle_tweets",
        batch_size: int = 16,
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

    def setup(self, stage: Optional[str] = None):
        data = KaggleTweets(data_dir=self.data_dir, train=True)
        n_val = int(len(data) * 0.2)
        n_train = len(data) - n_val
        self.train, self.val = random_split(data, [n_train, n_val])

        self.test = KaggleTweets(data_dir=self.data_dir, train=False)

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
