import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as TF
import torch
from pytorch_lightning.utilities.argparse import add_argparse_args, from_argparse_args

from transformers import BertModel


class BertClassifier(nn.Module):
    """Uses Hugging Face's BERT implementation to classify tweets."""

    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.fc = nn.Linear(768, 1)

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        return torch.sigmoid(self.fc(pooled_output))


class TweetClassifierModule(pl.LightningModule):
    """LightningModule implementation of BERT classifier for tweets."""

    def __init__(self, learning_rate: float = 1e-6):
        """
        Args:
            learning_rate (float, optional): Learning rate of optimiser. Defaults to 1e-6.
        """
        super().__init__()
        self.lr = learning_rate
        self.save_hyperparameters()
        self.net = BertClassifier()

    def forward(self, x):
        return self.net(x)

    def _step(self, batch, batch_idx):
        text_inputs, targets = batch
        mask = text_inputs["attention_mask"]
        input_id = text_inputs["input_ids"].squeeze(1)
        logits = self(input_id, mask)

        loss = F.cross_entropy(logits, targets)
        accuracy = TF.accuracy(logits, targets)

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
        text_inputs = batch  # no labels in competition test data
        mask = text_inputs["attention_mask"]
        input_id = text_inputs["input_ids"].squeeze(1)
        logits = self(input_id, mask)

        return text_inputs, logits > 0.5

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

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
