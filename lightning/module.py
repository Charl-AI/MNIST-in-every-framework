import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as TF


class MNISTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.layers(x)


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
        self.net = MNISTMLP()

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

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx)

    def validation_step_end(self, outs):
        self.log("validation/accuracy", outs["accuracy"])
        self.log("validation/loss", outs["loss"])

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        imgs = batch  # no labels in competition test data
        return imgs, torch.argmax(self(imgs), dim=1)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
