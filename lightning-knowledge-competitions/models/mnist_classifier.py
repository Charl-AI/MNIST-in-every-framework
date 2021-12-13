import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics.functional as TF
from pytorch_lightning.utilities.argparse import add_argparse_args, from_argparse_args

from models.base_models.resnet import ResNet50


class MNISTClassifierModule(pl.LightningModule):
    """LightningModule implementation of a ResNet for MNIST"""

    def __init__(self, learning_rate: float = 0.01):
        """
        Args:
            learning_rate (float, optional): Learning rate of optimiser. Defaults to 0.01.
        """
        super().__init__()
        self.lr = learning_rate
        self.save_hyperparameters()
        self.net = ResNet50(in_channels=1, out_classes=10)

    def forward(self, x):
        return self.net(x)

    def _step(self, batch, batch_idx):
        imgs, targets = batch
        logits = self(imgs)
        loss = F.cross_entropy(logits, targets.long())
        accuracy = TF.accuracy(logits, targets.int())

        return {
            "loss": loss,
            "accuracy": accuracy,
        }

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def training_step_end(self, outs):
        self.log("train/accuracy", outs["accuracy"])
        self.log("train/loss", outs["loss"])

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def validation_step_end(self, outs):
        self.log("validation/accuracy", outs["accuracy"])
        self.log("validation/loss", outs["loss"])

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        imgs = batch  # no labels in competition test data
        return imgs, torch.argmax(self(imgs), dim=1)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    @classmethod
    def add_argparse_args(cls, parent_parser, **kwargs):
        """Extends existing argparse by default `LightningDataModule` attributes.
        Uses the __init__ arguments and docstring to make the agument group."""
        return add_argparse_args(cls, parent_parser, **kwargs)

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        """Create an instance from CLI arguments.

        Args:
            args: The parser or namespace to take arguments from. Only known arguments will be
                parsed and passed to the :class:`~pytorch_lightning.core.datamodule.LightningDataModule`.
            **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
                These must be valid LightningModule arguments.
        """
        return from_argparse_args(cls, args, **kwargs)
