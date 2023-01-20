"""A ridiculously ugly implementation of an MLP with backprop in numpy.
You've seen the elegance of JAX, you've experienced the intuitiveness of PyTorch.
In less than 200 lines of code, we achieve neither."""

# this isn't really an autodiff library, but it works. If you have any suggestions
# to improve it, please keep them to yourself.

from typing import Sequence, Tuple
import numpy as np


class Linear:
    def __init__(self, in_features: int, out_features: int) -> None:
        self.w = np.random.randn(in_features, out_features)
        self.b = np.random.randn(out_features)

        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)

    def __call__(self, x: np.ndarray):
        return np.dot(x, self.w) + self.b

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        grad_x = np.dot(grad_y, self.w.T)
        self.grad_w = np.dot(grad_x.T, grad_y)
        self.grad_b = np.sum(grad_y, axis=0)
        return grad_x

    def zero_grad(self) -> None:
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)


class ReLU:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(x, 0)

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        return grad_y * (self.x > 0)


class Softmax:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        self.y = e_x / e_x.sum(axis=0)
        return self.y

    def backward(self, grad_y: np.ndarray) -> np.ndarray:
        return self.y * (grad_y - (grad_y * self.y).sum(axis=-1)[:, None])


class XEntropy:
    def __call__(self, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        # logits and targets are one-hot encoded
        self.logits = logits
        self.targets = targets
        return -np.sum(targets * np.log(logits + 1e-8)) / logits.shape[0]  # type: ignore

    def backward(self) -> np.ndarray:
        return (self.logits - self.targets) / self.logits.shape[0]


class SoftmaxXEntropy:
    def __call__(self, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        self.targets = targets
        logits_max = np.max(logits, axis=-1, keepdims=True)
        logits_exp = np.exp(logits - logits_max)
        logits_exp_sum = np.sum(logits_exp, axis=-1, keepdims=True)
        self.logits_softmax = logits_exp / logits_exp_sum
        loss = -np.sum(targets * np.log(self.logits_softmax + 1e-8), axis=-1)
        return loss.mean()

    def backward(self) -> np.ndarray:
        return self.logits_softmax - self.targets


class MLPClassifier:
    """MLP for multi-class classification. Uses softmax and cross-entropy loss."""

    def __init__(
        self,
        input_size: int = 784,
        layer_sizes: Sequence[int] = [512, 512],
        output_classes: int = 10,
    ) -> None:
        self.layers = []
        for output_size in layer_sizes:
            self.layers.append(Linear(input_size, output_size))
            self.layers.append(ReLU())
            input_size = output_size

        self.layers.append(Linear(input_size, output_classes))
        # self.layers.append(Softmax())

        # self.criterion = XEntropy()
        self.criterion = SoftmaxXEntropy()

        self.metric = lambda preds, targets: (  # targets are one-hot encoded
            preds.argmax(axis=-1) == targets.argmax(axis=-1)
        ).mean()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x

    def zero_grad(self) -> None:
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.zero_grad()

    def backward(self) -> None:
        grad_y = self.criterion.backward()
        for layer in reversed(self.layers):
            grad_y = layer.backward(grad_y)

    def train_step(
        self, data: np.ndarray, targets: np.ndarray, lr: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.zero_grad()
        logits = self(data)
        loss = self.criterion(logits, targets)
        self.backward()

        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.w -= lr * layer.grad_w
                layer.b -= lr * layer.grad_b

        acc = self.metric(logits, targets)
        return (loss, acc)

    def eval_step(
        self, data: np.ndarray, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        logits = self(data)
        loss = self.criterion(logits, targets)
        acc = self.metric(logits, targets)
        return (loss, acc)
