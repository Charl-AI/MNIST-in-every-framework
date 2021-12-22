# lightning-knowledge-competitions

Solutions to the Kaggle Knowledge Competitions in [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/).
Pytorch lightning is primarily aimed at researchers looking for the flexibility and comfortable API of Pytorch with a few
neat features for organising the research code and abstracting away engineering code.

Notice in particular how we don't need to write our own training loops. The logic of each training step is contained in the model 'LightningModule' which we pass to a trainer which is part of the library. The 'LightningDataModule' serves a similar purpose - it allows us to contain the logic for splitting and loading our dataset in one place. Also, the pre-made trainer has some features to save us time such as moving tensors between CPU and GPU automatically, mixed precision training, and some basic logging functionality.

## Training + Inference

The notebooks in ```lightning-knowledge-competitions/experiments/``` contain code for training the models and generating predictions. Pre-trained models are not provided in this repository due to their large size, but they can be easily reproduced by running the notebooks.

## Submission

It is easiest to submit results with the Kaggle API, for example:
```bash
# submits preds.csv to the mnist classification competition
kaggle competitions submit -c digit-recognizer -f data/kaggle_mnist/preds.csv --message first_submission_with_api
```
Each notebook will contain instructions for each individual competition.
