# lightning-knowledge-competitions

Solutions to the Kaggle Knowledge Competitions in [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/).
Pytorch lightning is primarily aimed at researchers looking for the flexibility and comfortable API of Pytorch with a few
neat features for organising the research code and abstracting away engineering code.

## Training

The notebooks in ```lightning-knowledge-competitions/experiments/``` contain all necessary code for training the final models. There are also training scripts with a CLI for training models directly - these are mostly useful for playing with hyperparameters. Show available options by using the ```--help``` argument with the training scripts. This will give a list of command line arguments you can use in the program. Pre-trained models are not provided in this repository due to their large size, but they can be easily reproduced by either running the notebooks or training scripts (with the appropriate hyperparameters).

This project integrates with [Weights and Biases](https://wandb.ai/site) for logging and it is strongly recommended to use it (it's free!). By default, including the ```--logger True``` flag in the CLI will use Weights and Biases.
When using Weights and Biases on a new machine, run ```wandb login``` in the terminal, and paste the API key from your weights and biases account when prompted to set it up.

## Inference, visualisation, submission

We use notebooks for inference and visualisation to improve readability. These can be found in ```lightning-knowledge-competitions/experiments/```. These notebooks will demonstrate how to use the trained models for prediction on the test dataset. Submission can be done through the kaggle API, for example:
```bash
# submits preds.csv to the mnist classification competition
kaggle competitions submit -c digit-recognizer -f data/kaggle_mnist/preds.csv --message first_submission_with_api
