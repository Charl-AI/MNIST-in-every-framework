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

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
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
What it does

This project uses PytorchLightning to organise the codebase and provide some useful abstractions.


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

For training, run:

```bash
python src/train.py
```

For options, run `python src/train.py --help` to show arguments you can pass from the command line.

### Citation
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```
