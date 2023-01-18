"""A ridiculously ugly implementation of an MLP with backprop in numpy.
You've seen the elegance of JAX, you've experienced the intuitiveness of PyTorch.
In less than 100 lines of code, we achieve neither."""

# this isn't really an autodiff library, but it works. If you have any suggestions
# to improve it, please keep them to yourself.

from typing import List, Sequence, Tuple
import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        self.w = np.random.randn(in_features, out_features)
        self.b = np.random.randn(out_features)

        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)

    def __call__(self, x):
        return np.dot(x, self.w) + self.b

    def backward(self, grad_y) -> np.ndarray:
        # grad wrt w and b are internally stored,
        # wrt x is returned to be used by the next layer

        grad_x = np.dot(grad_y, self.w.T)
        self.grad_w = np.dot(grad_x.T, grad_y)
        self.grad_b = np.sum(grad_y, axis=0)
        return grad_x


class ReLU:
    def __call__(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, grad_y):
        # no params, so just return the gradient wrt x
        return grad_y * (self.x > 0)


class MLP:
    def __init__(
        self, input_size: int = 784, output_sizes: Sequence[int] = [512, 512, 10]
    ) -> None:
        self.layers = []
        for output_size in output_sizes:
            self.layers.append(Linear(input_size, output_size))
            self.layers.append(ReLU())
            input_size = output_size

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad_y):
        for layer in reversed(self.layers):
            grad_y = layer.backward(grad_y)
        return grad_y

    def get_params(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        w = []
        b = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                w.append(layer.w)
                b.append(layer.b)
        return w, b

    def set_params(self, w: List[np.ndarray], b: List[np.ndarray]) -> None:
        for layer, weight, bias in zip(self.layers, w, b):
            if isinstance(layer, Linear):
                layer.w = weight
                layer.b = bias

    def gradients(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        grad_w = []
        grad_b = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                grad_w.append(layer.grad_w)
                grad_b.append(layer.grad_b)
        return grad_w, grad_b


class SigmoidXEntropy:
    def __call__(self, preds, targets) -> np.ndarray:
        self.preds = preds
        self.targets = targets
        return np.mean(-targets * np.log(preds) - (1 - targets) * np.log(1 - preds))

    def backward(self) -> np.ndarray:
        return (self.preds - self.targets) / (self.preds * (1 - self.preds) + 1e-8)
