import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import numpy as np
import optax


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
