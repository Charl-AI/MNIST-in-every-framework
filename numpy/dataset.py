"""Python (w/ numpy) code for downloading, extracting, and reading MNIST data.
We implement a miniature Pytorch-style dataset and dataloader too."""

import gzip
import math
import os
import struct
import urllib.request

import numpy as np

URL = "http://yann.lecun.com/exdb/mnist/"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def download_mnist(data_dir: str = "data/custom_mnist/") -> None:
    """Downloads MNIST to to data_dir if it doesn't already exist.

    Args:
        data_dir (str): Data directory for project. Defaults to "data/custom_mnist/".
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for file in FILES:
        if not os.path.exists(data_dir + file):
            print(f"Downloading {file} to {data_dir}")
            urllib.request.urlretrieve(URL + file, data_dir + file)
        else:
            print(f"File {file} already exists in {data_dir}, skipping download.")


def parse_images(
    data_dir: str = "data/custom_mnist/", train: bool = True
) -> np.ndarray:
    """Parses MNIST image file from data_dir into numpy array, shape (N, 28, 28).
    Reads 60k images from 'train-images-idx3-ubyte.gz' if train=True, else reads 10k
    from 't10k-images-idx3-ubyte.gz'.

    Args:
        data_dir (str): Data directory for project.
        train (bool): Whether to read training or test data. Defaults to True.

    Returns:
        np.ndarray: Numpy array of shape (N, 28, 28), dtype uint8.
    """
    filepath = (
        os.path.join(data_dir, "train-images-idx3-ubyte.gz")
        if train
        else os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    )

    with gzip.open(filepath, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder(">"))
        data = data.reshape((size, nrows, ncols))

    return data


def parse_labels(
    data_dir: str = "data/custom_mnist/", train: bool = True
) -> np.ndarray:
    """Parses MNIST label file from data_dir into numpy array, shape (N,).
    Reads 60k labels from 'train-labels-idx1-ubyte.gz' if train=True, else reads 10k
    from 't10k-labels-idx1-ubyte.gz'.

    Args:
        data_dir (str, optional): Data directory. Defaults to "data/custom_mnist/".
        train (bool, optional): Whether to read training or test data. Defaults to True.

    Returns:
        np.ndarray: Numpy array of shape (N,), dtype uint8.
    """
    filepath = (
        os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
        if train
        else os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    )

    with gzip.open(filepath, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder(">"))
        data = data.reshape((size,))

    return data


class MNIST:
    """Basic Pytorch-style dataset for MNIST."""

    def __init__(self, data_dir: str = "data/custom_mnist/", train: bool = True):
        self.data_dir = data_dir
        self.train = train
        self.images = parse_images(data_dir, train)
        self.labels = parse_labels(data_dir, train)

        self.images = np.reshape(self.images.astype(np.float32) / 255.0, (-1, 28 * 28))
        self.labels = self.labels.astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class MNISTLoader:
    """Basic Pytorch-style dataloader for MNIST."""

    def __init__(self, dataset: MNIST, batch_size: int, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i : i + self.batch_size]
            batch = [self.dataset[i] for i in batch_indices]
            images = np.stack([sample[0] for sample in batch])
            labels = np.stack([sample[1] for sample in batch])
            yield images, labels
