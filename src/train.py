from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

# import lightning module
# import datamodule

seed_everything(1)


def main(args):

    logger = TensorBoardLogger("runs")

    dict_args = vars(args)

    model = LightningModule(**dict_args)

    data = DataModule(batch_size=2)

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, data)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = LightningModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
