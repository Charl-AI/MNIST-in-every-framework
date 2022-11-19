import os
import logging
import pathlib
from dataclasses import dataclass

import jax
import optax
import tensorflow_datasets as tfds
import tensorflow as tf

import train
import model

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # JAX memory preallocation
tf.config.experimental.set_visible_devices([], "GPU")  # prevent TF/TB preallocation


@dataclass
class Configurations:
    rng: int
    log_dir: str = "outputs/"
    data_dir: str = "data/"
    batch_size: int = 128
    lr: float = 0.01
    num_epochs: int = 10
    val_split: float = 0.2


def main(cfg: Configurations):
    pathlib.Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=cfg.log_dir + "haiku.log",
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
        tf.reshape(tf.cast(x, tf.float32) / 255, (784,)),  # type: ignore
        y,
    )
    train_ds = (
        train_ds.map(preprocess_mnist)
        .batch(cfg.batch_size)
        .prefetch(1)
        .repeat(cfg.num_epochs)
    )
    val_ds = val_ds.map(preprocess_mnist).batch(cfg.batch_size).prefetch(1)
    test_ds = test_ds.map(preprocess_mnist).batch(cfg.batch_size).prefetch(1)

    rng = jax.random.PRNGKey(cfg.rng)
    dummy_batch = train.ImageBatch(*next(train_ds.as_numpy_iterator()))
    opt = optax.adam(cfg.lr)
    state = train.create_train_state(rng, dummy_batch, model.mnist_mlp, opt)

    state = train.train(
        state,
        train_ds,
        val_ds,
        model.mnist_mlp,
        opt,
        eval_every_n_steps=375,
        log_every_n_steps=100,
    )
    train.evaluate(state, test_ds, model.mnist_mlp, "test")


if __name__ == "__main__":
    cfg = Configurations(rng=42)
    main(cfg)
