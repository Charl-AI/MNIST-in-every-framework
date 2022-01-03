import haiku as hk
import jax.numpy as jnp
from haiku.nets import ResNet18


def _forward(
    imgs: jnp.ndarray,
    is_training: bool,
) -> jnp.ndarray:
    """Forward application of the resnet."""
    net = ResNet18(10, resnet_v2=False)
    return net(imgs, is_training=is_training)


# Transform our forwards function into a pair of pure functions.
forward = hk.transform_with_state(_forward)
