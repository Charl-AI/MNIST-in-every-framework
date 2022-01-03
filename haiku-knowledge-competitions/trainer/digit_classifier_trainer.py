from typing import Type, NamedTuple

import haiku as hk
import jax
import jmp
import jax.numpy as jnp
import numpy as np
import optax
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from models.digit_classifier import forward


class TrainState(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState
    loss_scale: jmp.LossScale
