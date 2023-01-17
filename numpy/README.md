# MNIST in numpy

NumPy is the premier library for scientific and numerical computing in Python. It is probably the biggest contributing factor towards Python's success in these communities. It is not used for deep learning because it does not run on GPU and does not have an autograd system. We won't worry about the GPU issue here (we only train a tiny classifier and don't use GPU at all in this repo), but we will implement a tiny custom autograd system.

We take inspiration from the brilliant [mattj/autodidact](https://github.com/mattjj/autodidact) and [geohot/tinygrad](https://github.com/geohot/tinygrad) for our autograd system. Ours is even smaller -- femtograd??
