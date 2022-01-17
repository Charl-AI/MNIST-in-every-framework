from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp


class MLP(nn.Module):
    """Flax implementation of an MLP with dropout."""

    features: Sequence[int]

    @nn.compact
    def __call__(self, x, train: bool = True):

        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
                x = nn.Dropout(0.2, deterministic=train)(x)
        return x
