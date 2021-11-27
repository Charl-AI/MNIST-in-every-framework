<div align="center">

# Kaggle-Knowledge-Competitions

[![Kaggle](http://img.shields.io/badge/Kaggle-CompetitionName-44c5ce.svg)](https://www.kaggle.com/competitions)
[![Blog](http://img.shields.io/badge/Blog-NameofPost-c044ce.svg)](https://charl-ai.github.io/)

</div>

## Executive Summary
This project solves Kaggle 'knowledge' competitions using Pytorch Lightning. These competitions are gentle introductions to different areas in machine learning and data science; this repository is both an introduction to Kaggle and an introduction to lightning. It's worth noting that these competitions seem to suffer from overfitting on the public leaderboards (for example, the MNIST competition has submissions with 100% accuracy!). I am not particularly interested in this extreme game of tuning/overfitting, so this repository will use standard methods that can give decent and robust results out of the box.

### 1. MNIST digit recogniser
We use a standard ResNet50 to perform image classification. The model gets 0.98246 accuracy on the test set after training for 5 epochs with the default arguments. This is pretty good, and you could probably push the accuracy higher with some more tuning.

### 2. House price regression
TODO

### 3. Titanic survival prediction
TODO

### 4. Flower Classification
TODO

### 5. Tweet NLP
TODO


This project uses [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) to organise the codebase and provide some useful abstractions.

## Installation
Note: this project has been developed and tested on Ubuntu. The project may run on other platforms, however, this is not guaranteed. If you do not have an Ubuntu machine, consider using WSL.

It is recommended to clone the repo with the GitHub CLI:
```bash
# clone project
gh repo clone Charl-AI/Kaggle-Knowledge-Competitions

# change to project directory
cd Kaggle-Knowledge-Competitions
```

A virtual environment is recommended for this project. Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt --no-cache-dir
```

If you are having CUDA issues, you can find the Pytorch builds for each version of CUDA [here](https://pytorch.org/get-started/locally/). For example, an NVIDIA RTX 3090 uses the CUDA 11.3 compute platform, this can be installed by running:

```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Data
The data used in this project is downloadable through the Kaggle API. By default, all models expect the data to be located in ```data/```, but this can be tweaked with the data_dir command line argument. To download the data using the Kaggle API, first ensure you have your Kaggle API key stored in ```~/.kaggle/kaggle.json```, then run the following:

```bash
# create and enter data directory
mkdir data
cd data

# download datasets
kaggle competitions download -p kaggle_mnist -c digit-recognizer

# prepare datasets
unzip kaggle_mnist/digit-recognizer.zip -d kaggle_mnist
```


## Training

For training, show available options by running:

```bash
python src/train.py --help
```

This will give a list of command line arguments you can use in the program - e.g. ```python src/train.py --max_epochs 50 --batch_size 8 --log_every_n_steps 5 --learning_rate 0.01```.

This project integrates with [Weights and Biases](https://wandb.ai/site) for logging and it is strongly recommended to use it (it's free!). By default, including the ```--logger True``` flag in the CLI will use Weights and Biases.
When using Weights and Biases on a new machine, run ```wandb login``` in the terminal, and paste the API key from your weights and biases account when prompted to set it up.

## Inference, visualisation, submission

We use notebooks for inference and visualisation to improve readability. These can be found in ```src/experiments/```. These notebooks will demonstrate how to use the trained models for prediction on the test dataset. Submission can be done through the kaggle API, for example:
```bash
# submits preds.csv to the mnist classification competition
kaggle competitions submit -c digit-recognizer -f data/kaggle_mnist/preds.csv --message first_submission_with_api
```
