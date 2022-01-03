import os
from typing import Optional
import pandas as pd
import jax.numpy as jnp
from torch.utils.data import Dataset


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
            self.labels = jnp.array(df["label"].values)
            self.imgs = df.drop(labels="label", axis=1)
            self.imgs = jnp.array(self.imgs.values)

        else:
            df = pd.read_csv(os.path.join(data_dir, "test.csv"))
            self.imgs = jnp.array(df.values)

        self.imgs = jnp.reshape(self.imgs, (-1, 28, 28))
        self.imgs = jnp.expand_dims(self.imgs, axis=1)
        self.imgs = jnp.float32(self.imgs) / 255.0

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
