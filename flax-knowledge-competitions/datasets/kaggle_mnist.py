import os
from typing import Optional
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader


class KaggleMNIST(Dataset):
    """Dataset class for MNIST downloaded from Kaggle API. We use Pytorch's
    Dataset and Dataloader classes because JAX doesn't come with its own.
    The key difference is that we use numpy arrays instead of torch tensors."""

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
            self.labels = np.array(df["label"].values)
            self.imgs = df.drop(labels="label", axis=1)
            self.imgs = np.array(self.imgs.values)

        else:
            df = pd.read_csv(os.path.join(data_dir, "test.csv"))
            self.imgs = np.array(df.values)

        self.imgs = np.reshape(self.imgs, (-1, 28, 28))
        self.imgs = np.expand_dims(self.imgs, axis=-1)
        self.imgs = np.float32(self.imgs) / 255.0

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


def _numpy_collate(batch):
    """Utility function for dataloding numpy arrays into JAX."""
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [_numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(DataLoader):
    """Shim of Pytorch DataLoader that loads numpy arrays into JAX."""

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=_numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
