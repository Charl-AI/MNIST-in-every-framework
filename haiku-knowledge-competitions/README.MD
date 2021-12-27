# jax-knowledge-competitions

Solutions to the Kaggle Knowledge Competitions in [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html), with [Haiku](https://dm-haiku.readthedocs.io/en/latest/notebooks/basics.html). As great as JAX is, it is quite low level, so it is desirable to have some higher level libraries for deep learning. Haiku, from DeepMind, is one of the most popular JAX libraries (Flax is also popular) - it provides Pytorch-like 'Modules' to manage state whilst maintaining JAX's functional nature. We also use Optax, another DeepMind library, to manage optimisers.

add extra paragraphs about structure of project and how it differs to the Pytorch and Pure JAX projects.

## Training + Inference

The notebooks in `experiments/` contain code for training the models and generating predictions. Pre-trained models are not provided in this repository due to their large size, but they can be easily reproduced by running the notebooks.

## Submission

It is easiest to submit results with the Kaggle API, for example:
```bash
# submits preds.csv to the mnist classification competition
kaggle competitions submit -c digit-recognizer -f data/kaggle_mnist/preds.csv --message first_submission_with_api
```
Each notebook will contain instructions for each individual competition.
