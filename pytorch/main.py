import logging
import pathlib
from dataclasses import dataclass

import torch
import torchvision
from torch.utils.data import DataLoader, random_split

from model import MNISTMLP
from trainer import test_digit_classifier, train_digit_classifier


@dataclass
class Configurations:
    data_dir: str = "data/"
    batch_size: int = 128
    num_workers: int = 4
    lr: float = 0.01
    num_epochs: int = 5
    device: str = "cuda"
    log_every_n_steps: int = 50
    val_split: float = 0.2
    log_dir: str = "outputs/"


def main(cfg: Configurations):
    pathlib.Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=cfg.log_dir + "pytorch.log",
        filemode="w",
        level=logging.INFO,
        encoding="utf-8",
    )
    device = torch.device(cfg.device)
    model = MNISTMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr)
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    data = torchvision.datasets.MNIST(
        root=cfg.data_dir,
        download=True,
        train=True,
        transform=transforms,
    )

    n_val = int(len(data) * cfg.val_split)
    n_train = len(data) - n_val
    train_data, val_data = random_split(data, [n_train, n_val])

    train_loader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    val_loader = DataLoader(
        val_data,
        shuffle=False,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    train_digit_classifier(
        model,
        train_loader,
        val_loader,
        num_epochs=cfg.num_epochs,
        device=device,
        optimizer=optimizer,
        log_every_n_steps=cfg.log_every_n_steps,
    )
    test_data = torchvision.datasets.MNIST(
        download=True, root=cfg.data_dir, train=False, transform=transforms
    )
    test_loader = DataLoader(
        test_data,
        shuffle=False,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    test_digit_classifier(model, test_loader, device)


if __name__ == "__main__":
    cfg = Configurations()
    main(cfg)
