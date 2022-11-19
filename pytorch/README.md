# MNIST in PyTorch

Pytorch is a popular framework for deep learning and is particularly widespread in research. It has a stable API and good documentation which makes it great for developers. In Pytorch, we typically need to define three things: our model, our dataset, and our training loop. PyTorch has a very large ecosystem, including torchvision, which we use here to help with the model and dataset.

Run the model with: `python pytorch/main.py`. The outputs are logged with the inbuilt Python logger to `outputs/pytorch.log`. For more serious projects, consider using Hydra for configuring the model and TensorBoard for logging.
