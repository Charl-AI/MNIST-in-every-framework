<div align="center">

# Kaggle-Knowledge-Competitions

[![Kaggle](http://img.shields.io/badge/Kaggle-Competitions-44c5ce.svg)](https://www.kaggle.com/competitions)
[![Blog](http://img.shields.io/badge/Blog-TBD-c044ce.svg)](https://charl-ai.github.io/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


</div>

## Executive Summary

This project aims so solve Kaggle 'getting started' competitions with a variety of different deep learning frameworks and libraries. The main focus is to implement relatively standard deep learning models in each framework to get a better understanding of their respective strengths and weaknesses. Each framework is a 'mini-project', with its own directory containing the code for all of the competitions. The table below shows the competitions included, as well as the models and frameworks used.

|                   | Digit Classifier (ResNet18 ~98% accuracy) | Sales Forecasting (LSTM ~?% RMSLE) | Art Generation (StyleGAN3 ~? FID) | Titanic Survival (Bootstrapped ensemble MLP ~?% Accuracy)| Tweet NLP (BERT ~80% F1) |
|-------------------|-----------------------------|--------------------------|-----------------------------|--------------------------|--------------------------|
| PyTorch           | <ul><li>- [x] </li></ul>    | <ul><li>- [ ] </li></ul> | <ul><li>- [ ] </li></ul>    | <ul><li>- [ ] </li></ul> | <ul><li>- [ ] </li></ul> |
| PyTorch Lightning | <ul><li>- [x] </li></ul>    | <ul><li>- [ ] </li></ul> | <ul><li>- [ ] </li></ul>    | <ul><li>- [ ] </li></ul> | <ul><li>- [x] </li></ul> |
| TensorFlow        | <ul><li>- [ ] </li></ul>    | <ul><li>- [ ] </li></ul> | <ul><li>- [ ] </li></ul>    | <ul><li>- [ ] </li></ul> | <ul><li>- [ ] </li></ul> |
| JAX-Flax              | <ul><li>- [ ] </li></ul>    | <ul><li>- [ ] </li></ul> | <ul><li>- [ ] </li></ul>    | <ul><li>- [ ] </li></ul> | <ul><li>- [ ] </li></ul> |
| JAX-Haiku             | <ul><li>- [ ] </li></ul>    | <ul><li>- [ ] </li></ul> | <ul><li>- [ ] </li></ul>    | <ul><li>- [ ] </li></ul> | <ul><li>- [ ] </li></ul> |

You might notice that I have only given the rough score that each method attains, not specific scores for each method/framework combination. This project is really about getting proficient in each of the deep learning libraries, not scoring highly on the competitions. It is also worth noting that these competitions seem to suffer from overfitting on the public leaderboards - for example, the MNIST competition has submissions with 100% accuracy! I am not interested in this extreme game of tuning/overfitting as I don't think there is much to be learned by doing so; this is also why I avoid diving into feature engineering, XGBoost, or traditional ML methods in this project (even though they are all good methods for these competitions). This is not a data science project, it is about practicing coding in different styles and frameworks.


## Installation

This project includes both a `requirements.txt` file, as well as a `Dockerfile` and `devcontainer.json`. This enables two methods for installation.

### Method 1: devcontainers (recommended for full reproduction of development environment)

If you have Docker and VScode (with the remote development extension pack) installed, you can reproduce the entire development environment including OS, Python version, CUDA version, and dependencies by simply running `Remote containers: Clone Repository in Container Volume` from the command palette (alternatively, you could clone the repository and run `Remote Containers: Open folder in Container`). This is the easiest way to install the project. If you use Docker but don't like VScode, feel free to try building from the Dockerfile, although some small changes might be necessary.

**This method requires GPU drivers capable of CUDA 11.3 (check this by ensuring CUDA Version >= 11.3 when you run `nvidia-smi`)**

### Method 2: python virtual environments (recommended if you do not have a CUDA >= 11.3 capable GPU or if you do not use Docker)

Clone the repository, then create, activate, and install dependencies in a [Python virtual environment](https://docs.python.org/3/tutorial/venv.html) in the usual way. **Ensure you are using Python 3.8 - this is what the project is built on**.

Depending on your CUDA driver capabilities / CUDA toolkit version, you may have to replace the deep learning libraries with versions suited to your setup. Instructions can be found here for [PyTorch](https://pytorch.org/get-started/locally/), [JAX](https://github.com/google/jax#installation), and [TensorFlow](https://www.tensorflow.org/install/gpu).


## Data

The data used in this project is downloadable through the Kaggle API. By default, all models expect the data to be located in ```data/```, but this can usually be changed if necessary. To download the data using the Kaggle API, first ensure you have your Kaggle API key stored in ```~/.kaggle/kaggle.json```, then run the included shell script, ensuring you are running it from the project root (i.e. the directory containing this README):

```bash
bash ./download_data.sh
```

## Running

Each mini-project has its own README, describing how to properly do training, inference, visualisation, and submission of results. Generally, this project tries to use notebooks in place of `main.py` files to enable easy visualisation of results. A nice benefit is that the notebook outputs get uploaded to GitHub, so you can see the outputs of the code without downloading and running it.
