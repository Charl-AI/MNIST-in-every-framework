from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything

from model.resnet import MNISTResNetModule
from datasets.MNIST_kaggle import MNISTKaggleModule

seed_everything(1)


def main(args):

    dict_args = vars(args)

    if args.model_name == "mnist":
        model = MNISTResNetModule(**dict_args)
        data = MNISTKaggleModule(batch_size=32)
    # TODO: implement other models
    else:
        raise Exception("Unknown model name.")

    trainer = pl.Trainer.from_argparse_args(args)

    if dict_args["auto_lr_find"] == True:
        trainer.tune(model, data)

    trainer.fit(model, data)


if __name__ == "__main__":

    parser = ArgumentParser()

    # figure out which model to use
    parser.add_argument(
        "--model_name", type=str, default="mnist", help="mnist, titanic, or housing"
    )

    parser = pl.Trainer.add_argparse_args(parser)

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # let the model add what it wants
    if temp_args.model_name == "mnist":
        parser = MNISTResNetModule.add_model_specific_args(parser)

    args = parser.parse_args()

    main(args)
