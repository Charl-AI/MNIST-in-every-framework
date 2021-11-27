import os
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything

from datasets.kaggle_mnist import KaggleMNISTDataModule
from models.mnist_classifier import MNISTClassifierModule
from visualisation.wandb_callbacks.image_classification_callback import (
    WandbImageClassificationCallback,
)

# dataloader workers get different seeds to prevent augmentations being repeated
seed_everything(1, workers=True)


def main(args):

    model = MNISTClassifierModule(lr=args.learning_rate)
    data = KaggleMNISTDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=args.drop_last,
    )

    callbacks = []

    if args.logger:
        # Get name of project root. Assumes structure of root/src/train.py
        root_name = os.path.basename(Path(__file__).resolve().parent.parent)

        logger = WandbLogger(log_model=False, project=f"{root_name}-logs")
        logger.watch(model)

        callbacks.extend(
            [
                WandbImageClassificationCallback(data, num_samples=32),
            ]
        )
    else:
        logger = False

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, data)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = MNISTClassifierModule.add_argparse_args(parser)
    parser = KaggleMNISTDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
