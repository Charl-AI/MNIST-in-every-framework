from typing import Optional
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader, random_split


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/kaggle_mnist",
        batch_size: int = 50,
        num_workers: int = 4,
        drop_last: bool = False,
    ) -> None:
        """
        Args:
            data_dir: path to the data directory
            batch_size: batch size for dataloaders
            num_workers: number of workers for dataloaders, usually 4*num GPUs is fine,
                set to zero if you find a memory leak.
            drop_last: whether to drop the last batch to keep batch sizes constant
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last

        self.transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.dims = (1, 28, 28)

    def setup(self, stage: Optional[str] = None):
        data = torchvision.datasets.MNIST(
            root=self.data_dir,
            download=True,
            train=True,
            transform=self.transforms,
        )
        n_val = int(len(data) * 0.2)
        n_train = len(data) - n_val
        self.train, self.val = random_split(data, [n_train, n_val])

        self.test = torchvision.datasets.MNIST(
            root=self.data_dir,
            download=True,
            train=False,
            transform=self.transforms,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )
