# MNIST in TensorFlow

We use [TensorFlow](https://www.tensorflow.org/) (2), with [Keras](https://keras.io/). TensorFlow was one of the first successful deep learning frameworks and Keras is an API specification for interacting with frameworks at a high-level of abstraction. TensorFlow went through an overhaul to create TF2, which improved the API and made the Keras integration tighter. Nowadays, TensorFlow is popular in industry and is the easiest framework to productionise. In research, PyTorch and JAX are generally preferred.

Notice that, like Pytorch Lightning, we don't need to write any training loops because Keras takes care of that for us. Run the model with: `python tensorflow/main.py`. The outputs are logged with the inbuilt Python logger to `outputs/tensorflow.log`. For more serious projects, consider using Hydra for configuring the model and TensorBoard for logging.
