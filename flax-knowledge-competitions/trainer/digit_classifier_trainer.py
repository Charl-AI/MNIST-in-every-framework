from typing import Type
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


def cross_entropy_loss(logits, labels, num_classes=10):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=num_classes)
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    return jnp.mean(xentropy)


def compute_metrics(logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics


def create_train_state(net, rng, learning_rate, momentum):
    """Creates initial `TrainState`."""
    params = net.init(rng, jnp.ones([1, 1, 28, 28]))["params"]
    tx = optax.sgd(learning_rate, momentum)
    return TrainState.create(apply_fn=net.apply, params=params, tx=tx)


@jax.jit
def train_step(state, net, batch):
    """Train for a single step."""
    imgs, targets = batch

    def loss_fn(params):
        logits = net.apply({"params": params}, imgs)
        loss = cross_entropy_loss(logits=logits, labels=targets)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=targets)
    return state, metrics


@jax.jit
def eval_step(params, net, batch):
    imgs, targets = batch
    logits = net.apply({"params": params}, imgs)
    return compute_metrics(logits=logits, labels=targets)


def run_epoch(
    state: TrainState,
    net: Type[nn.Module],
    loader: DataLoader,
    epoch: int,
    mode: str,
    logger: SummaryWriter = None,
    log_every_n_steps: int = 100,
):
    assert mode in ["train", "val"], f"Mode must be either 'train' or 'val', got {mode}"
    assert (
        log_every_n_steps < len(loader) or logger is None
    ), f"log_every_n_steps must be less than the length of the dataloader, got {log_every_n_steps} >= {len(loader)} for {mode} mode"

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
        if mode == "train":
            state, metrics = train_step(state, net, (imgs, targets))
        else:
            metrics = eval_step(state.params, net, (imgs, targets))
        total_loss += metrics["loss"]
        total_accuracy += metrics["accuracy"]

        if logger is not None and (batch_idx + 1) % log_every_n_steps == 0:
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

    return state


def train_digit_classifier(
    net: Type[nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    init_rng: jax.random.PRNGKey,
    learning_rate: float,
    momentum: float,
    num_epochs: int,
    logger: SummaryWriter = None,
    log_every_n_steps: int = 100,
):
    """Train the digit classifier model.

    Args:
        net (Type[nn.Module]): Model to train.
        train_loader (DataLoader): Dataloader of training data.
        val_loader (DataLoader): Dataloader of validation data.
        init_rng (jax.random.PRNGKey): PRNG key for parameter initialisation.
        learning_rate (float): Learning rate for SGD optimiser.
        momentum (float): Momentum for SGD optimiser.
        num_epochs (int): Number of epochs to train for.
        logger (SummaryWriter, optional): Tensorboard logger to use. Defaults to None.
        log_every_n_steps (int, optional): Log metrics to tensorboard every n steps. Defaults to 100.
    """
    state = create_train_state(init_rng, learning_rate, momentum)
    del init_rng  # Must not be used anymore.

    for epoch in range(num_epochs):
        for mode in ["train", "val"]:
            loader = train_loader if mode == "train" else val_loader

            state = run_epoch(
                state,
                net,
                loader,
                epoch,
                mode,
                logger=logger,
                log_every_n_steps=log_every_n_steps,
            )
    return state


def test_digit_classifier(
    net: Type[nn.Module],
    state: TrainState,
    test_loader: DataLoader,
):
    """Test the digit classifier on unlabeled data and return the predictions.

    Args:
        net (Type[nn.Module]): Model to test.
        state (TrainState): State of the model.
        test_loader (DataLoader): Dataloader of test data (should not have labels).

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

        logits = net.apply({"params": state.params}, imgs)
        preds.append((imgs, jnp.argmax(logits, -1)))

    return preds
