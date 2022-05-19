import os
from typing import Tuple
import tensorflow as tf
import pandas as pd
import numpy as np


def create_dataset(data_dir: str, train: bool = True) -> tf.data.Dataset:
    """Create tf.data.Dataset object for Kaggle MNIST dataset. Performs preprocessing
    by normalizing and reshaping the data.

    Args:
        data_dir (str): Path to directory containing train.csv and test.csv.
        train (bool, optional): If true, reads from train.csv, else test.csv (test data
        does not have labels). Defaults to True.

    Returns:
        tf.data.Dataset: tensorflow dataset object for Kaggle MNIST dataset.
    """

    if train:
        df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        labels = np.array(df["label"].values)
        labels = tf.one_hot(labels, 10)
        imgs = df.drop(labels="label", axis=1)
        imgs = np.array(imgs.values)
    else:
        df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        imgs = np.array(df.values)

    imgs = np.reshape(imgs, (-1, 28, 28))
    imgs = np.expand_dims(imgs, axis=-1)
    imgs = np.float32(imgs) / 255.0

    # resize to 32x32 because our resnet needs this as minimum size
    imgs = tf.image.resize(imgs, (32, 32))

    if train:
        return tf.data.Dataset.from_tensor_slices((imgs, labels))  # type: ignore
    else:
        return tf.data.Dataset.from_tensor_slices(imgs)


def get_batched_datasets(
    data_dir: str, batch_size: int = 32, val_split: float = 0.2
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Get preprocessed train, validation, and test datasets for Kaggle MNIST dataset.

    Args:
        data_dir (str): Path to directory containing train.csv and test.csv.
        batch_size (int, optional): Batch size. Defaults to 32.
        val_split (float, optional): fraction reserved for validation. Defaults to 0.2.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Tuple of train,
        validation, and test datasets. (with batching, prefetching, and caching).
    """

    train_val_ds = create_dataset(data_dir, train=True)
    test_ds = create_dataset(data_dir, train=False)

    len_val_ds = int(len(train_val_ds) * val_split)
    len_train_ds = len(train_val_ds) - len_val_ds

    train_ds = train_val_ds.take(len_train_ds)
    val_ds = train_val_ds.skip(len_train_ds)

    train_ds = train_ds.shuffle(buffer_size=len(train_ds))

    batch_prefetch_cache = (
        lambda d: d.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
        .cache()
    )
    train_ds = batch_prefetch_cache(train_ds)
    val_ds = batch_prefetch_cache(val_ds)
    test_ds = batch_prefetch_cache(test_ds)

    return train_ds, val_ds, test_ds
