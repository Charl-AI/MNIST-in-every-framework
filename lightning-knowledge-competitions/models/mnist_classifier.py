import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics.functional as TF
from pytorch_lightning.utilities.argparse import add_argparse_args, from_argparse_args
from torchvision.models.resnet import BasicBlock, ResNet


class ResNet18(ResNet):
    """ResNet 18, based on torchvision implementation [BSD 3-Clause License].
    Modified to allow for different numbers of input channels (e.g grayscale).

    If you want a pretrained model, use the official torchvision implementation
    (pretrained models only exist for 3-channel inputs, so the channel
    modifications made here would be useless anyway).

    Input to forward method: Image Tensor, size [Bx in_channels xHxW]
    Output of forward method: Predictions Tensor, size [Bx num_classes]

    """

    def __init__(self, in_channels: int = 3, out_classes: int = 10):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=out_classes)

        # simply change the first layer to accept the number of input channels
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )


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
        self.net = ResNet18(in_channels=1, out_classes=10)

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
