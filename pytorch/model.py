import torch
from torchvision.models.resnet import BasicBlock, ResNet


class ResNet18(ResNet):
    """ResNet 18, based on torchvision implementation [BSD 3-Clause License].
    Modified to allow for different numbers of input channels (e.g grayscale).

    If you want a pretrained model, use the official torchvision implementation
    (pretrained models only exist for 3-channel inputs, so the channel
    modifications made here would be useless anyway).

    Input to forward method: Image Tensor, size [Bx in_channels xHxW]
    Output of forward method: Predictions Tensor, size [Bx num_classes]

    """

    def __init__(self, in_channels: int = 3, out_classes: int = 10) -> None:
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=out_classes)

        # simply change the first layer to accept the number of input channels
        self.conv1 = torch.nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
