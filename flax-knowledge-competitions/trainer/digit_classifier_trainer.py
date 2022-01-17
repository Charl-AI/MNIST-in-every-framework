from functools import partial
from multiprocessing import dummy
from sqlite3 import paramstyle
from typing import Callable, Iterable, Mapping, NamedTuple, Tuple, Any

import flax
import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import jmp
import optax
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

Scalars = Mapping[str, jnp.ndarray]


class TrainState(train_state.TrainState):
    batch_stats: Any = None


def create_train_state(
    rng: int,
    net: nn.Module,
    batch_shape: jnp.ndarray,
    optimizer: optax.GradientTransformation,
) -> TrainState:
    """Creates initial `TrainState`
    Args:
        rng (int): Seed for PRNG key for parameter initialization.
        net (nn.Module): Network to train.
        batch_shape (jnp.ndarray): Shape of batched inputs e.g. [B, C, H, W].
        optimizer (optax.GradientTransformation): Optax optimizer to use.
    Returns:
        TrainState: Flax class for storing training state and useful methods.
    """
    rng = jax.random.PRNGKey(rng)
    dummy_input = jnp.ones(batch_shape, dtype=net.dtype)
    jit_init = jax.jit(net.init)
    variables = jit_init({"params": rng}, dummy_input)
    params = variables["params"]
    batch_stats = variables["batch_stats"]
    return TrainState.create(
        apply_fn=net.apply, params=params, tx=optimizer, batch_stats=batch_stats
    )


def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    return jnp.mean(xentropy)


def compute_metrics(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
) -> Scalars:
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics


@jax.jit
def train_step(state, batch):
    imgs, labels = batch

    def loss_fn(params):
        """loss function used for training."""
        variables = {"params": params, "batch_stats": state.batch_stats}
        logits, new_model_state = state.apply_fn(
            variables,
            imgs,
            mutable=["batch_stats"],
        )
        loss = cross_entropy_loss(logits, labels)
        return loss, (new_model_state, logits)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    new_model_state, logits = aux[1]
    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state["batch_stats"]
    )
    metrics = compute_metrics(logits, labels)
    return new_state, metrics


@jax.jit
def val_step(state, batch):
    imgs, labels = batch
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    logits = state.apply_fn(variables, imgs, train=False, mutable=False)
    return compute_metrics(logits, labels)


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
    min_loader_len = min(len(train_loader), len(val_loader))
    assert log_every_n_steps < min_loader_len or logger is None, (
        "log_every_n_steps must be less than the length of the dataloader, got"
        f" {log_every_n_steps=} >= {min_loader_len=}"
    )

    imgs, *_ = next(iter(train_loader))  # dummy batch for init
    batch_shape = imgs.shape
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
                        total_loss / log_every_n_steps,
                        global_step + step,
                    )
                    logger.add_scalar(
                        f"{mode}/accuracy",
                        total_accuracy / log_every_n_steps,
                        global_step + step,
                    )
                    total_loss = 0
                    total_accuracy = 0
    return state
