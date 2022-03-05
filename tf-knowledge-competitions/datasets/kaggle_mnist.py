import os
import tensorflow as tf
import pandas as pd
import numpy as np


def create_dataset(data_dir: str, train: bool = True):

    if train:
        df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        labels = np.array(df["label"].values)
        imgs = df.drop(labels="label", axis=1)
        imgs = np.array(imgs.values)
    else:
        df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        imgs = np.array(df.values)

    imgs = np.reshape(imgs, (-1, 28, 28))
    imgs = np.expand_dims(imgs, axis=-1)
    imgs = np.float32(imgs) / 255.0

    if train:
        return tf.data.Dataset.from_tensor_slices((imgs, labels))
    else:
        return tf.data.Dataset.from_tensor_slices(imgs)
