from tensorflow.keras.applications.resnet import ResNet
from tensorflow.keras.applications.resnet import stack1


def ResNet18(
    include_top=True,
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    **kwargs
):
    """Instantiates the ResNet18 architecture. Just a shim of the Keras ResNet code
    because they don't provide the 18 variant out of the box."""

    weights = None  # No pretrained weights provided by Keras

    def stack_fn(x):
        x = stack1(x, 64, 2, stride1=1, name="conv2")
        x = stack1(x, 128, 2, name="conv3")
        x = stack1(x, 256, 2, name="conv4")
        return stack1(x, 512, 2, name="conv5")

    return ResNet(
        stack_fn,
        False,
        True,
        "resnet18",
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        **kwargs
    )
