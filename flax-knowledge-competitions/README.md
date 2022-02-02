# flax-knowledge-competitions

Solutions to the Kaggle Knowledge Competitions in [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html), with [Flax](https://flax.readthedocs.io/en/latest/overview.html). JAX is gaining popularity in research - it offers a functional, low-level API and is self-described as 'NumPy on the CPU, GPU, and TPU, with great automatic differentiation for high-performance machine learning research'. It is quite low-level, so usually higher level libraries are used on top of it. In this case, we use Flax and Optax, which are generally the most mature libraries in the JAX ecosystem.

The fundamental principle of JAX is that pure functions can be easily compiled, vectorised, and parallelised with `jit()`, `vmap()`, and `pmap()`, respectively. This makes it more performant because `jit()` can optimise your programs specifically, instead of relying on prebuilt CUDA kernels which sacrifice some performance to be more general. In Flax, neural networks are reduced to a pair of pure functions, an init function which initialises the parameters, and an apply function which takes an input and the parameters and produces the output. Because the parameters are no longer part of the network state, they must be passed around from function to function - FLAX provides a `TrainState` class which helps with this and also has some helper methods.

Purifying stateful networks may not seem important, but I like how it reduces 'spooky action at a distance' and makes it clear exactly what is going on in training. JAX also has a nice way of handling RNG which should help with determinism and reproducibility.

One final thing to note is that JAX currently has an experimental feature called `xmap()` which can be seen as an arbitrary combination of `vmap()` and `pmap()`. This, combined with the named tensors feature (also experimental) could allow future models to be written for a single example and trivially batched and parallelised across N devices with a single context manager.

TODO: implement mixed precision
