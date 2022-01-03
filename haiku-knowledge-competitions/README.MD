# haiku-knowledge-competitions

Solutions to the Kaggle Knowledge Competitions in [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html), with [Haiku](https://dm-haiku.readthedocs.io/en/latest/notebooks/basics.html). JAX is gaining popularity in research - it offers a functional, low-level API and is self-described as 'NumPy on the CPU, GPU, and TPU, with great automatic differentiation for high-performance machine learning research'. It is quite low-level, so it is useful to have higher-level neural network libraries such as Flax or Haiku on top of it. We also use Optax, from DeepMind, to manage optimisers.

add extra paragraphs about structure of project and how it differs to the Pytorch and JAX-FLAX projects.

## Training + Inference

The notebooks in `experiments/` contain code for training the models and generating predictions. Pre-trained models are not provided in this repository due to their large size, but they can be easily reproduced by running the notebooks.

## Submission

It is easiest to submit results with the Kaggle API, for example:
```bash
# submits preds.csv to the mnist classification competition
kaggle competitions submit -c digit-recognizer -f data/kaggle_mnist/preds.csv --message first_submission_with_api
```
Each notebook will contain instructions for each individual competition.
