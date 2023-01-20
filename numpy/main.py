import os
import logging
import pathlib
from dataclasses import dataclass

import train
import nn
import dataset


@dataclass
class Configurations:
    log_dir: str = "outputs/"
    data_dir: str = "data/custom_mnist"
    batch_size: int = 64
    lr: float = 0.0001
    num_epochs: int = 10
    train_split: float = 0.8


def main(cfg: Configurations):
    pathlib.Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=cfg.log_dir + "numpy.log",
        filemode="w",
        level=logging.INFO,
        encoding="utf-8",
    )

    test_ds = dataset.MNIST(cfg.data_dir, train=False)
    train_ds, val_ds = dataset.create_train_val_split(
        dataset.MNIST(cfg.data_dir, train=True), cfg.train_split
    )
    train_loader = dataset.MNISTLoader(
        train_ds, cfg.batch_size, num_epochs=cfg.num_epochs
    )
    val_loader = dataset.MNISTLoader(val_ds, cfg.batch_size, num_epochs=1)
    test_loader = dataset.MNISTLoader(test_ds, cfg.batch_size, num_epochs=1)

    model = nn.MLPClassifier()
    model = train.train(model, train_loader, val_loader, cfg.lr)
    train.evaluate(model, test_loader, "test")


if __name__ == "__main__":
    main(Configurations())
