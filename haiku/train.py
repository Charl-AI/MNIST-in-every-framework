import functools
from typing import Literal, NamedTuple, Tuple, Mapping
import optax
import haiku as hk
import jax
from tqdm import tqdm
import tensorflow as tf
import jax.numpy as jnp
import logging


class ImageBatch(NamedTuple):
    """A batch of image data contains images (BHWC) and labels (B)."""

    image: jnp.ndarray
    label: jnp.ndarray


class TrainState(NamedTuple):
    """TrainState is a container for storing model params and optimizer state."""

    global_step: int
    params: optax.Params
    opt_state: optax.OptState


def create_train_state(
    rng: jax.random.KeyArray,
    dummy_batch: ImageBatch,
    model_fwd: hk.Transformed,
    tx: optax.GradientTransformation,
) -> TrainState:
    """Create the initial TrainState by initialising params and optimizer."""

    params = model_fwd.init(rng, dummy_batch.image)
    opt_state = tx.init(params)
    return TrainState(global_step=0, params=params, opt_state=opt_state)


@functools.partial(jax.jit, static_argnums=(2, 3))
def train_step(
    state: TrainState,
    batch: ImageBatch,
    model_fwd: hk.Transformed,
    tx: optax.GradientTransformation,
) -> Tuple[TrainState, Mapping[str, float]]:
    """A training step takes a batch of data and the current state and returns the new
    state and a dictionary of metrics for the batch."""

    images, labels = batch

    def loss_fn(params):
        logits = model_fwd.apply(params, images)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    updates, new_opt_state = tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    metrics = {"loss": loss, "accuracy": jnp.mean(jnp.argmax(logits, -1) == labels)}
    return (
        TrainState(
            global_step=state.global_step + 1,
            params=new_params,
            opt_state=new_opt_state,
        ),
        metrics,
    )


@functools.partial(jax.jit, static_argnums=(2,))
def eval_step(
    state: TrainState,
    batch: ImageBatch,
    model_fwd: hk.Transformed,
) -> Mapping[str, float]:
    """Evaluate the model over a batch of data, returning a dictionary of metrics."""

    images, labels = batch

    logits = model_fwd.apply(state.params, images)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
    metrics = {"loss": loss, "accuracy": jnp.mean(jnp.argmax(logits, -1) == labels)}
    return metrics


def evaluate(
    state: TrainState,
    eval_ds: tf.data.Dataset,
    model_fwd: hk.Transformed,
    mode: Literal["val", "test"],
) -> None:
    """Evaluate the model by running through the val/test set for one epoch. Metrics
    are logged to the logger and nothing is returned."""

    eval_metrics = []
    for batch in tqdm(
        eval_ds.as_numpy_iterator(), total=len(eval_ds), unit="step", desc="Evaluating"
    ):
        batch = ImageBatch(*batch)  # can use typeguard+duck typing here instead
        metrics = eval_step(state, batch, model_fwd)
        eval_metrics.append(metrics)

    eval_metrics = jax.tree_util.tree_map(
        lambda *ms: jnp.mean(jnp.array(ms)), *eval_metrics
    )
    logging.info(f"{mode} metrics: {eval_metrics}")
    return None


def train(
    state: TrainState,
    train_ds: tf.data.Dataset,
    eval_ds: tf.data.Dataset,
    model_fwd: hk.Transformed,
    tx: optax.GradientTransformation,
    log_every_n_steps: int = 100,
    eval_every_n_steps: int = 1000,
) -> TrainState:
    """Train the model, logging training and evaluation metrics. Note: train_ds, should
    already be repeated for the desired number of epochs i.e. with train_ds.repeat(n)"""

    train_metrics = []
    for batch in tqdm(
        train_ds.as_numpy_iterator(), total=len(train_ds), unit="step", desc="Training"
    ):
        batch = ImageBatch(*batch)
        state, metrics = train_step(state, batch, model_fwd, tx)
        train_metrics.append(metrics)

        if state.global_step % log_every_n_steps == 0:
            train_metrics = jax.tree_util.tree_map(
                lambda *ms: jnp.mean(jnp.array(ms)), *train_metrics
            )
            logging.info(f"train metrics: {train_metrics}")
            train_metrics = []

        if state.global_step % eval_every_n_steps == 0:
            _ = evaluate(state, eval_ds, model_fwd, mode="val")

    return state
