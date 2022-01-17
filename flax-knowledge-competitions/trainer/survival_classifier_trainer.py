from typing import Any, Mapping

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

Scalars = Mapping[str, jnp.ndarray]


def create_train_state(
    rng: jax.random.PRNGKey,
    net: nn.Module,
    batch_shape: jnp.ndarray,
    optimizer: optax.GradientTransformation,
) -> TrainState:
    """Creates initial `TrainState`
    Args:
        rng (jax.random.PRNGKey): PRNG key for parameter initialization.
        net (nn.Module): Network to train.
        batch_shape (jnp.ndarray): Shape of batched inputs e.g. [B, C, H, W].
        optimizer (optax.GradientTransformation): Optax optimizer to use.
    Returns:
        TrainState: Flax class for storing training state and useful methods.
    """
    dummy_input = jnp.ones(batch_shape, dtype=net.dtype)
    jit_init = jax.jit(net.init)
    params = jit_init({"params": rng}, dummy_input)
    return TrainState.create(apply_fn=net.apply, params=params, tx=optimizer)


def binary_cross_entropy_loss(logits, labels):
    xentropy = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels)
    return jnp.mean(xentropy)


def compute_metrics(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
) -> Scalars:
    loss = binary_cross_entropy_loss(logits, labels)
    # threshold at 0.5 (not necessarily optimal)
    preds = jnp.where(jax.nn.sigmoid(logits) < 0.5, 0, 1)
    accuracy = jnp.mean(preds == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics


@jax.jit
def train_step(state, batch):
    inputs, targets = batch

    def loss_fn(params):
        """loss function used for training."""
        logits = state.apply_fn(
            params,
            inputs,
        )
        loss = binary_cross_entropy_loss(logits, targets)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    logits = aux[1]
    new_state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, targets)
    return new_state, metrics


@jax.jit
def val_step(state, batch):
    inputs, targets = batch
    logits = state.apply_fn(state.params, inputs, train=False)
    return compute_metrics(logits, targets)


@jax.jit
def test_step(state, batch):
    inputs = batch
    logits = state.apply_fn(state.params, inputs, train=False)

    # threshold at 0.5 (not necessarily optimal)
    preds = jnp.where(jax.nn.sigmoid(logits) < 0.5, 0, 1)
    return inputs, preds


def train_digit_classifier(
    rng: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    optimizer: optax.GradientTransformation,
    logger: SummaryWriter = None,
    log_every_n_steps: int = 100,
):
    """Trains the digit classifier.

    Args:
        rng (int): Seed for PRNG key for parameter initialization.
        model (nn.Module): Flax Network to train.
        train_loader (DataLoader): Pytorch DataLoader for training data.
        val_loader (DataLoader): Pytorch DataLoader for validation data.
        num_epochs (int): Number of epochs to train for.
        optimizer (optax.GradientTransformation): Optax optimizer to use.
        logger (SummaryWriter, optional): TensorBoard logger to use. Defaults to None.
        log_every_n_steps (int, optional): How often to log to TB. Defaults to 100.

    Returns:
        TrainState: Final state of the network.
    """
    min_loader_len = min(len(train_loader), len(val_loader))
    assert log_every_n_steps < min_loader_len or logger is None, (
        "log_every_n_steps must be less than the length of the dataloader, got"
        f" {log_every_n_steps=} >= {min_loader_len=}"
    )

    inputs, *_ = next(iter(train_loader))  # dummy batch for init
    batch_shape = inputs.shape
    state = create_train_state(rng, model, batch_shape, optimizer)

    for epoch in range(num_epochs):
        for mode in ["train", "val"]:
            loader = train_loader if mode == "train" else val_loader

            global_step = epoch * len(loader)
            total_accuracy = 0.0
            total_loss = 0.0

            for step, batch in enumerate(
                tqdm(loader, total=len(loader), desc=f"Epoch {epoch}, {mode}")
            ):
                if mode == "train":
                    state, metrics = train_step(state, batch)
                else:
                    metrics = val_step(state, batch)

                total_accuracy += metrics["accuracy"]
                total_loss += metrics["loss"]

                if logger is not None and (step + 1) % log_every_n_steps == 0:
                    # mean the loss and accuracy over the last n batches
                    logger.add_scalar(
                        f"{mode}/loss",
                        np.array(total_loss / log_every_n_steps),
                        global_step + step,
                    )
                    logger.add_scalar(
                        f"{mode}/accuracy",
                        np.array(total_accuracy / log_every_n_steps),
                        global_step + step,
                    )
                    total_loss = 0
                    total_accuracy = 0
    return state


def test_digit_classifier(state: TrainState, test_loader: DataLoader):
    """Generate an array of predictions for the test set.

    Args:
        state (TrainState): State of the network.
        test_loader (DataLoader): Pytorch DataLoader for test data.

    Returns:
        np.ndarray: Array of predictions for the test set.
    """

    preds = []
    inputs = []

    for batch in tqdm(
        test_loader, total=len(test_loader), desc="Generating Predictions on Test Set"
    ):
        batch_inputs, batch_preds = test_step(state, batch)
        preds.append(batch_preds)
        inputs.append(batch_inputs)

    final_preds = np.array(jnp.concatenate(preds, axis=0))
    final_inputs = np.array(jnp.concatenate(inputs, axis=0))
    return final_inputs, final_preds
