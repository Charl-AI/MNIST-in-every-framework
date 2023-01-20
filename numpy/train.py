from typing import Literal

import nn
import dataset

import numpy as np
import logging
from tqdm import tqdm


def evaluate(
    model: nn.MLPClassifier,
    loader: dataset.MNISTLoader,
    mode: Literal["val", "test"],
) -> None:
    eval_loss = []
    eval_acc = []

    for imgs, labels in tqdm(loader, total=len(loader), unit="step", desc="Evaluating"):
        batch_loss, batch_acc = model.eval_step(imgs, labels)

        eval_loss.append(batch_loss)
        eval_acc.append(batch_acc)

    eval_loss = np.mean(np.array(eval_loss))
    eval_acc = np.mean(np.array(eval_acc))
    logging.info(f"{mode} metrics: {{'loss': {eval_loss}, 'accuracy': {eval_acc}}}")


def train(
    model: nn.MLPClassifier,
    train_loader: dataset.MNISTLoader,
    val_loader: dataset.MNISTLoader,
    lr: float,
    log_every_n_steps: int = 100,
    eval_every_n_steps: int = 1000,
) -> nn.MLPClassifier:
    loss = []
    acc = []

    for step, (imgs, labels) in enumerate(
        tqdm(train_loader, total=len(train_loader), unit="step", desc="Training")
    ):
        batch_loss, batch_acc = model.train_step(imgs, labels, lr)
        loss.append(batch_loss)
        acc.append(batch_acc)

        if (step + 1) % log_every_n_steps == 0:
            loss = np.mean(np.array(loss))
            acc = np.mean(np.array(acc))
            logging.info(f"train metrics: {{'loss': {loss}, 'accuracy': {acc}}}")
            loss = []
            acc = []

        if (step + 1) % eval_every_n_steps == 0:
            evaluate(model, val_loader, mode="val")

    return model
