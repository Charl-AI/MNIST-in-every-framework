#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="Kaggle-Knowledge-Competitions",
    version="0.0.0",
    description="Solution to the three knowledge competitions on Kaggle. Implemented in Pytorch-Lightning.",
    author="C Jones",
    author_email="",
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url="https://github.com/Charl-AI/Kaggle-Knowledge-Competitions",
    install_requires=["pytorch-lightning"],
    packages=find_packages(),
)
