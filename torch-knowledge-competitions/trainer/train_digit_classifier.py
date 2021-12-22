import torch
from tqdm import tqdm


def train_digit_classifier(
    model,
    train_loader,
    val_loader,
    num_epochs,
    optimizer,
    device,
):

    for epoch in range(num_epochs):
        for mode in ["train", "val"]:

            loader = train_loader if mode == "train" else val_loader

            for batch in tqdm(
                loader,
                total=len(loader),
                desc=mode,
            ):
                pass


def predict_digit_classifier():
    pass
