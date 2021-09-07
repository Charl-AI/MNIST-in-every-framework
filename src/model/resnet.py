from torchvision.models.resnet import ResNet, BasicBlock
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics


class _MNISTResNet(ResNet):
    """Pytorch ResNet 18, adapted for use with MNIST"""

    def __init__(self):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)

        # simply change the first layer to accept greyscale
        self.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )


class MNISTResNetModule(pl.LightningModule):
    """LightningModule implementation of a ResNet for MNIST"""

    def __init__(self, lr: float = 0.01, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.train_accuracy = torchmetrics.Accuracy()
        self.valid_accuracy = torchmetrics.Accuracy()
        self.net = _MNISTResNet()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        predictions = self(imgs)

        loss = F.cross_entropy(predictions, labels)

        self.log_dict(
            {"train_loss": loss, "train_acc": self.train_accuracy(predictions, labels)}
        )

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        predictions = self(imgs)

        loss = F.cross_entropy(predictions, labels)

        self.log_dict(
            {"val_loss": loss, "val_acc": self.valid_accuracy(predictions, labels)}
        )

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("StochasticUnetSystem")
        parser.add_argument(
            "-lr",
            "--learning_rate",
            help="learning rate of optimiser",
            type=float,
            default=0.01,
        )
        return parent_parser
