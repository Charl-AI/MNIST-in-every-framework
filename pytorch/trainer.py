import logging
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    mode: Literal["train", "val", "test"],
    log_every_n_steps: int = 100,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    """Run a single epoch of training or evaluation. Results are logged to standard
    output (INFO level).

    Args:
        model (Type[nn.Module]): Model to train/evaluate.
        loader (DataLoader): Dataloader of training/evaluation data.
        device (torch.device): Device to run the model on.
        epoch (int): Current epoch number.
        mode (str): "train", "val", "test" string to specify training or evaluation mode.
        optimizer (Type[torch.optim.Optimizer]): optimizer to use (training mode only).
            Defaults to None.
        log_every_n_steps (int): Log metrics every n steps.
    """
    assert mode in [
        "train",
        "val",
        "test",
    ], f"Mode must be either 'train', 'val', 'test, got {mode}"
    assert log_every_n_steps < len(loader), (
        "log_every_n_steps must be less than the length of the dataloader, got"
        f" {log_every_n_steps} >= {len(loader)} for {mode} mode"
    )

    model.train() if mode == "train" else model.eval()
    global_step = epoch * len(loader)  # total number of steps taken so far
    total_loss = 0
    total_accuracy = 0

    for batch_idx, (imgs, targets) in enumerate(
        tqdm(
            loader,
            total=len(loader),
            desc=f"{mode} epoch {epoch}: ",
        )
    ):
        optimizer.zero_grad() if optimizer else None

        imgs, targets = imgs.to(device), targets.to(device)
        logits = model(imgs)
        loss = F.cross_entropy(logits, targets.long())
        accuracy = TF.accuracy(logits, targets.int())

        total_loss += loss.item()
        total_accuracy += accuracy.item()

        if mode == "train":
            loss.backward()
            optimizer.step() if optimizer else None

        if (batch_idx + 1) % log_every_n_steps == 0:
            # mean the loss and accuracy over the last n batches
            logging.info(
                f"{mode}/loss, step {global_step + batch_idx}:"
                f" {total_loss / log_every_n_steps}"
            )
            logging.info(
                f"{mode}/accuracy, step {global_step + batch_idx}:"
                f" {total_accuracy / log_every_n_steps}"
            )
            total_loss = 0
            total_accuracy = 0


def train_digit_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
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
        profiler (torch.profiler.profile, optional): Profiler to use. Defaults to None.
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
                    mode,  # type: ignore
                    optimizer=optimizer,
                    log_every_n_steps=log_every_n_steps,
                )


def test_digit_classifier(
    model: nn.Module, test_loader: DataLoader, device: torch.device
) -> None:
    """Test the digit classifier on the test set, logging the results.

    Args:
        model (Type[nn.Module]): Model to test.
        test_loader (DataLoader): Dataloader of test data (should not have labels).
        device (torch.device): Device to test on.
    """
    model = model.to(device)
    with torch.set_grad_enabled(False):
        run_epoch(
            model,
            test_loader,
            device,
            epoch=0,
            mode="test",
            optimizer=None,
            log_every_n_steps=len(test_loader) - 1,
        )
