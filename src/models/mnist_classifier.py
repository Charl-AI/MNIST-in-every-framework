import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics.functional as TF

from models.base_models.resnet import ResNet50


class MNISTClassifierModule(pl.LightningModule):
    """LightningModule implementation of a ResNet for MNIST"""

    def __init__(self, lr: float):
        super().__init__()
        self.lr = lr
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

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    @classmethod
    def add_argparse_args(cls, parent_parser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--learning_rate",
            help="learning rate of optimiser",
            type=float,
            default=0.01,
        )
        return parent_parser
