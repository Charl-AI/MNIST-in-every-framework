# jax-knowledge-competitions

Solutions to the Kaggle Knowledge Competitions in [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html). JAX is gaining popularity in research - it offers a functional, low-level API and is self-described as 'NumPy on the CPU, GPU, and TPU, with great automatic differentiation for high-performance machine learning research'. We use Stax and Optix from the `jax.experimental` library to help build the deep learning models.

add extra paragraphs about structure of project and how JAX differs to Pytorch

## Training + Inference

The notebooks in `experiments/` contain code for training the models and generating predictions. Pre-trained models are not provided in this repository due to their large size, but they can be easily reproduced by running the notebooks.

## Submission

It is easiest to submit results with the Kaggle API, for example:
```bash
# submits preds.csv to the mnist classification competition
kaggle competitions submit -c digit-recognizer -f data/kaggle_mnist/preds.csv --message first_submission_with_api
```
Each notebook will contain instructions for each individual competition.
