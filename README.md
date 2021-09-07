Checklist for starting project:

* [ ] Create First Working Version Milestone (accomplished when this checklist is complete)
* [ ] Add custom issue labels (for example severity of issue, (more) issue types, and staus of issue (abandoned, in progress, etc.)
* [ ] Update setup.py
* [ ] Update README (readme driven development)

* [ ] Perform EDA and update first cell of notebook
* [ ] (Optional) Create proof of concept in notebook (using toy data etc...)
* [ ] Create datamodule (example one for testing is fine)
* [ ] Create lightningmodule (basic one for testing is fine)
* [ ] Update train.py (must be working with argparsing and logging)


<div align="center">

# TITLE

[![Competitions](http://img.shields.io/badge/Kaggle-4b44ce.svg)](https://www.kaggle.com/competitions)
<!--
ARXIV
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--
Conference
-->
</div>

## Description
This project solves the three Kaggle knowledge competitions using Pytorch Lightning. It is both an introduction to Kaggle and an introduction to lightning.

### 1. MNIST digit recogniser


### 2. House price regression


### 3. Titanic survival prediction


## Installation
First, clone the repo
```bash
# clone project
git clone https://github.com/Charl-AI/REPO-NAME

# change to project directory
cd deep-learning-project-template
```

A virtual environment is recommended for this project. Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv venv
source venv/bin/activate

pip install -e .
pip install -r requirements.txt
```

## Running

For training the models, run:

```bash
python src/train.py
```
You can use the ```model_name``` flag to determine which model you want to train, with options for each of the competitions:

```bash
--mnist
--prices
--titanic
```

For more options, run `python src/train.py --help` to show arguments you can pass from the command line.
