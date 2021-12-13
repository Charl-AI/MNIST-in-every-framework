<div align="center">

# Kaggle-Knowledge-Competitions

[![Kaggle](http://img.shields.io/badge/Kaggle-Competitions-44c5ce.svg)](https://www.kaggle.com/competitions)
[![Blog](http://img.shields.io/badge/Blog-TBD-c044ce.svg)](https://charl-ai.github.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


</div>

## Executive Summary

This project aims so solve each of the Kaggle 'knowledge' competitions with a variety of different deep learning frameworks and libraries. The main focus is to implement relatively standard deep learning
models in each framework to get a better understanding of their respective strengths and weaknesses. Each framework is a 'mini-project', with its own directory containing the code for all of the competitions. The table below shows the progress made on each pairing of framework and competition task.

|                   | Digit Classifier (ResNet18) | House Price Regression | Titanic Survival Prediction | Flower Classification | Tweet NLP |
|-------------------|-----------------------------|------------------------|-----------------------------|-----------------------|-----------|
| Pytorch           | <ul><li>- [ ] </li><li>                       | - [ ]                  | - [ ]                       | - [ ]                 | - [ ]     |
| Pytorch Lightning | <ul><li>- [x] </li><li>    | - [ ]                  | - [ ]                       | - [ ]                 | - [ ]     |
| TensorFlow        | - [ ]                       | - [ ]                  | - [ ]                       | - [ ]                 | - [ ]     |
| JAX               | - [ ]                       | - [ ]                  | - [ ]                       | - [ ]                 | - [ ]     |
| Haiku             | - [ ]                       | - [ ]                  | - [ ]                       | - [ ]                 | - [ ]     |

Note: these competitions are gentle introductions to different areas of machine learning and this repository focusses mainly as an introduction to the most popular deep learning frameworks. It's worth noting that these competitions seem to suffer from overfitting on the public leaderboards (for example, the MNIST competition has submissions with 100% accuracy!). I am not particularly interested in this extreme game of tuning/overfitting, so this repository will focus on implementing standard methods that can give decent and robust results out of the box.


## Installation

This project includes both a `requirements.txt` file, as well as a `Dockerfile` and `devcontainer.json`. This enables two methods for installation:

### Method 1: devcontainers (recommended for full reproducibility of development environment)

Devcontainers allow for full reproducibility of the OS, python version, and dependencies. This can help to avoid getting stuck in dependency hell by running everything inside a container with pinned versions.
This project integrates with the VScode remote containers extension - if you have this set up, you can simply run `Remote containers: Clone Repository in Container Volume` from the command
palette to set up this project - you can also clone the repository normally and use the `Remote Containers: Open folder in Container` command.
If you have Docker installed but do not wish to use VScode, you can build the container from the included Dockerfile.

Caveat: Currently the container will install the version of Pytorch compatible with CUDA 11.3 (featured on cards such as the RTX 3090). If you do not have a card capable of this, you will
need to install the correct version of Pytorch from [here](https://pytorch.org/get-started/locally/). In future, I will design the container to do this automatically for you.

### Method 2: python virtual environments (higher chance something goes wrong, but more familiar to most researchers and requires no docker installation)

First, clone the repo, it is recommended to use the GitHub CLI:
```bash
# clone project
gh repo clone Charl-AI/Kaggle-Knowledge-Competitions

# change to project directory
cd Kaggle-Knowledge-Competitions
```

Create and activate a virtual environment, then install the dependencies:

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

The data used in this project is downloadable through the Kaggle API. By default, all models expect the data to be located in ```data/``` by default, but this can usually be changed if necessary. To download the data using the Kaggle API, first ensure you have your Kaggle API key stored in ```~/.kaggle/kaggle.json```, then run the following:

```bash
# create and enter data directory
mkdir data
cd data

# download datasets
kaggle competitions download -p kaggle_mnist -c digit-recognizer

# prepare datasets
unzip kaggle_mnist/digit-recognizer.zip -d kaggle_mnist
```

## Running

Each mini-project has its own README, describing how to properly do training, inference, visualisation, and submission of results.
