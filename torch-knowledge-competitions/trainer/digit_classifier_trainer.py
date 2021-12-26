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

            for imgs, targets in tqdm(
                loader,
                total=len(loader),
                desc=mode,
            ):
                optimizer.zero_grad()

                imgs, targets = imgs.to(device), targets.to(device)

                loss.backward()
                optimizer.step()


def predict_digit_classifier():
    pass
