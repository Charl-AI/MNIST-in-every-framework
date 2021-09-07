from src.datasets.MNIST_kaggle import MNISTKaggle
from src.datasets.MNIST_kaggle import MNISTKaggleModule
from tests.decorators import local_test


@local_test
def test_mnist_train_dataset():
    data = MNISTKaggle(train=True)
    assert data[0][0].shape == (1, 28, 28)


@local_test
def test_mnist_datamodule():
    datamodule = MNISTKaggleModule(batch_size=2)
    datamodule.setup(stage="fit")
    batch = next(iter(datamodule.train_dataloader()))
    assert batch[0].shape == (2, 1, 28, 28)


def remote_test():
    assert True
