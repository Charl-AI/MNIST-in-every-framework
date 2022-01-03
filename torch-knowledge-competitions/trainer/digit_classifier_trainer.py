from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as TF
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


def run_epoch(
    model: Type[nn.Module],
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    mode: str,
    optimizer: Type[torch.optim.Optimizer] = None,
    logger: SummaryWriter = None,
    log_every_n_steps: int = 100,
) -> None:
    """Run a single epoch of training or evaluation. Results are logged to tensorboard.

    Args:
        model (Type[nn.Module]): Model to train/evaluate.
        loader (DataLoader): Dataloader of training/evaluation data.
        device (torch.device): Device to run the model on.
        epoch (int): Current epoch number.
        mode (str): "train" or "val" string to specify training or evaluation mode.
        optimizer (Type[torch.optim.Optimizer]): optimizer to use (training mode only).
            Defaults to None.
        logger (SummaryWriter): Tensorboard logger to use.
        log_every_n_steps (int): Log metrics to tensorboard every n steps.
    """
    assert mode in ["train", "val"], f"Mode must be either 'train' or 'val', got {mode}"
    assert (
        log_every_n_steps < len(loader) or logger is None
    ), f"log_every_n_steps must be less than the length of the dataloader, got {log_every_n_steps} >= {len(loader)} for {mode} mode"

    model.train() if mode == "train" else model.eval()
    global_step = epoch * len(loader)  # total number of steps taken so far
    total_loss = 0
    total_accuracy = 0

    for batch_idx, (imgs, targets) in enumerate(
        tqdm(
            loader,
            total=len(loader),
            desc=f"Epoch {epoch}: {mode}",
        )
    ):
        optimizer.zero_grad() if mode == "train" else None

        imgs, targets = imgs.to(device), targets.to(device)
        logits = model(imgs)
        loss = F.cross_entropy(logits, targets.long())
        accuracy = TF.accuracy(logits, targets.int())

        total_loss += loss.item()
        total_accuracy += accuracy.item()

        if mode == "train":
            loss.backward()
            optimizer.step()

        if logger is not None and batch_idx + 1 % log_every_n_steps == 0:
            # mean the loss and accuracy over the last n batches
            logger.add_scalar(
                f"{mode}/loss", total_loss / log_every_n_steps, global_step + batch_idx
            )
            logger.add_scalar(
                f"{mode}/accuracy",
                total_accuracy / log_every_n_steps,
                global_step + batch_idx,
            )
            total_loss = 0
            total_accuracy = 0


def train_digit_classifier(
    model: Type[nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    optimizer: Type[torch.optim.Optimizer] = None,
    logger: SummaryWriter = None,
    log_every_n_steps: int = 100,
):
    """Train the digit classifier model.

    Args:
        model (Type[nn.Module]): Model to train.
        train_loader (DataLoader): Dataloader of training data.
        val_loader (DataLoader): Dataloader of validation data.
        num_epochs (int): Number of epochs to train for.
        device (torch.device): Device to run the model on.
        optimizer (Type[torch.optim.Optimizer], optional): Optimizer to use. Defaults to None.
        logger (SummaryWriter, optional): Tensorboard logger to use. Defaults to None.
        log_every_n_steps (int, optional): Log metrics to tensorboard every n steps. Defaults to 100.
    """
    model = model.to(device)
    for epoch in range(num_epochs):
        for mode in ["train", "val"]:
            loader = train_loader if mode == "train" else val_loader

            # equivalent to using torch.no_grad() during validation
            with torch.set_grad_enabled(mode == "train"):
                run_epoch(
                    model,
                    loader,
                    device,
                    epoch,
                    mode,
                    optimizer=optimizer,
                    logger=logger,
                    log_every_n_steps=log_every_n_steps,
                )


def test_digit_classifier(
    model: Type[nn.Module], test_loader: DataLoader, device: torch.device
):
    """Test the digit classifier on unlabeled data and return the predictions.

    Args:
        model (Type[nn.Module]): Model to test.
        test_loader (DataLoader): Dataloader of test data (should not have labels).
        device (torch.device): Device to test on.

    Returns:
        preds (List of Tuples of Tensors [(imgs, preds)...]): Predictions for the test
            data. List of length len(test_loader). Containing a tuple of tensors, with
            each tensor length of batch_size.
    """
    preds = []

    for batch_idx, imgs in enumerate(
        tqdm(
            test_loader,
            total=len(test_loader),
            desc=f"Generating predictions on test set",
        )
    ):

        imgs = imgs.to(device)
        logits = model(imgs)
        preds.append((imgs, torch.argmax(logits, dim=1)))

    return preds
