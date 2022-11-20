import logging
import pathlib
from dataclasses import dataclass

from module import MNISTClassifierModule
from dataset import MNISTDataModule
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers


@dataclass
class Configurations:
    data_dir: str = "data/"
    batch_size: int = 128
    num_workers: int = 4
    lr: float = 0.01
    num_epochs: int = 5
    log_every_n_steps: int = 50
    val_split: float = 0.2
    log_dir: str = "outputs/"


def main(cfg: Configurations):
    logger = pl_loggers.CSVLogger(cfg.log_dir, name="lightning")
    model = MNISTClassifierModule(learning_rate=cfg.lr)
    data = MNISTDataModule(cfg.data_dir, cfg.batch_size, cfg.num_workers)
    trainer = pl.Trainer(
        default_root_dir=cfg.log_dir,
        max_epochs=cfg.num_epochs,
        precision=16,
        enable_checkpointing=False,
        logger=logger,
        log_every_n_steps=cfg.log_every_n_steps,
    )
    trainer.fit(model, data)
    trainer.test(model, data)


if __name__ == "__main__":
    cfg = Configurations()
    main(cfg)
