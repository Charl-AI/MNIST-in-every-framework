import logging
import pathlib
from dataclasses import dataclass

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

from model import ResNet18


@dataclass
class Configurations:
    log_dir: str = "outputs/"
    data_dir: str = "data/"
    batch_size: int = 128
    lr: float = 0.01
    num_epochs: int = 5
    val_split: float = 0.2


def main(cfg: Configurations):
    pathlib.Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=cfg.log_dir + "tensorflow.log",
        filemode="w",
        level=logging.INFO,
        encoding="utf-8",
    )
    train_percent = int((1 - cfg.val_split) * 100)
    train_ds, val_ds, test_ds = tfds.load(  # type: ignore
        "mnist",
        split=[f"train[:{train_percent}%]", f"train[{train_percent}%:]", "test"],
        as_supervised=True,
        data_dir=cfg.data_dir,
    )
    assert isinstance(train_ds, tf.data.Dataset)  # just to make mypy happy
    assert isinstance(val_ds, tf.data.Dataset)
    assert isinstance(test_ds, tf.data.Dataset)

    preprocess_mnist = lambda x, y: (
        tf.image.resize((tf.cast(x, tf.float32) / 255.0), (32, 32)),  # type: ignore
        tf.one_hot(y, 10),
    )
    train_ds = train_ds.map(preprocess_mnist).batch(cfg.batch_size).prefetch(1)
    val_ds = val_ds.map(preprocess_mnist).batch(cfg.batch_size).prefetch(1)
    test_ds = test_ds.map(preprocess_mnist).batch(cfg.batch_size).prefetch(1)

    model = ResNet18(classes=10, include_top=True, input_shape=(32, 32, 1))
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=cfg.lr),
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[keras.metrics.CategoricalAccuracy()],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.num_epochs,
    )
    model.evaluate(test_ds)


if __name__ == "__main__":
    cfg = Configurations()
    main(cfg)
