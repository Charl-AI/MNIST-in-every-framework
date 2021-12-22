import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchmetrics import Accuracy, MetricCollection, F1
from pytorch_lightning.utilities.argparse import add_argparse_args, from_argparse_args

from transformers import BertModel


class BertClassifier(nn.Module):
    """Uses Hugging Face's BERT implementation to classify tweets."""

    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.fc = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        return self.fc(pooled_output)


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

        metrics = MetricCollection([Accuracy(), F1()])
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="validation/")

    def forward(self, input_id, mask):
        return self.net(input_id, mask)

    def training_step(self, batch, batch_idx):
        text_inputs, targets = batch
        mask = text_inputs["attention_mask"]
        input_id = text_inputs["input_ids"].squeeze(1)
        logits = self(input_id, mask).squeeze(1)

        loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        metrics = self.train_metrics(torch.sigmoid(logits), targets)
        self.log("train/loss", loss)
        self.log_dict(metrics)

        return loss

    def validation_step(self, batch, batch_idx):
        text_inputs, targets = batch
        mask = text_inputs["attention_mask"]
        input_id = text_inputs["input_ids"].squeeze(1)
        logits = self(input_id, mask).squeeze(1)

        loss = F.binary_cross_entropy_with_logits(logits, targets.float())
        metrics = self.val_metrics(torch.sigmoid(logits), targets)
        self.log("validation/loss", loss)
        self.log_dict(metrics)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        text_inputs = batch  # no labels in competition test data
        mask = text_inputs["attention_mask"]
        input_id = text_inputs["input_ids"].squeeze(1)
        preds = torch.sigmoid(self(input_id, mask).squeeze(1))

        return torch.where(preds > 0.5, torch.ones_like(preds), torch.zeros_like(preds))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

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
