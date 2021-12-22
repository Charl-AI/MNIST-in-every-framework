# torch-knowledge-competitions

Solutions to the Kaggle Knowledge Competitions in [Pytorch](https://pytorch.org/).
Pytorch is a popular framework for deep learning and is particularly widespread in research. It has a stable API and good documentation which makes it great for developers.

In Pytorch, we typically need to define three things: our model, our dataset, and our training loop. These are contained in the `models/`, `datasets/`, and `trainer/` files respectively. The flexibility of Pytorch is one of its best features, however, it sometimes leaves you writing somewhat repetitive code (for example, most training loops will look pretty similar).

## Training + Inference

The notebooks in `experiments/` contain code for training the models and generating predictions. Pre-trained models are not provided in this repository due to their large size, but they can be easily reproduced by running the notebooks.

## Submission

It is easiest to submit results with the Kaggle API, for example:
```bash
# submits preds.csv to the mnist classification competition
kaggle competitions submit -c digit-recognizer -f data/kaggle_mnist/preds.csv --message first_submission_with_api
```
Each notebook will contain instructions for each individual competition.
