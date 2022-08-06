<div align="center">

# MNIST in Every Framework

</div>

## Summary

*I hire not the researcher who has trained 10000 models once, I hire the researcher who has trained 1 model 10000 times* - Bruce Lee, probably

We implement and train a GPU-accelerated ResNet18 MNIST classifier in every (deep learning adjacent) framework and language. The goal is to practice coding in different languages and frameworks.

|                   | Implemented? | Run command: |
|-------------------|-----------------------------|--------------------------|
| PyTorch           | <ul><li>- [x] </li></ul>    | `bash pytorch/run.sh` |
| PyTorch-Lightning | <ul><li>- [x] </li></ul>    | `bash pytorch-lightning/run.sh` |
| TensorFlow        | <ul><li>- [x] </li></ul>    | `bash tensorflow/run.sh` |
| JAX + Haiku       | <ul><li>- [x] </li></ul>    | `bash jax/run.sh` |
| Julia             | <ul><li>- [ ] </li></ul>    | `bash julia/run.sh` |
| Rust              | <ul><li>- [ ] </li></ul>    | `bash rust/run.sh` |
| Haskell           | <ul><li>- [ ] </li></ul>    | `bash haskell/run.sh` |


## Why do all of this?

This project started because I wanted to learn JAX and thought it would be a good idea to have a repository of several problems implemented in both PyTorch and JAX to compare the different styles. As I started adding more frameworks to compare, it became too much of a burden to maintain all of the problems in all of the frameworks so I decided to reduce the scope to only focus on MNIST. It turns out that MNIST classification is a great project for learning new languages and frameworks for a number of reasons:

1. Familiar. Everyone knows what a MNIST ResNet project looks like. This helps you understand what the code is doing, even in a new language.
2. Ecosystem. Most major Python frameworks have a substantial ecosystem with lots of image classification examples. MNIST classification is thus a great introduction to things you will use in practice like `tf.data` or `torch.nn`.
3. Lack of ecosystem. Outside of Python, there is much more to implement yourself. This is a great way to improve at the language because you will have to learn how to do things like implementing dataloaders and layers etc...


## Installation and Running

We provide a Dockerfile for installation and running the project. The image requires your host machine to have a GPU + driver capable of running CUDA 11.3 (you can check this by running `nvidia-smi` and looking at the 'CUDA Version' field). If you don't, you will have to install everything manually.

### 0. Prerequisites

Clone this repo. Next, install Docker with the NVIDIA container runtime. Instructions can be found [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

### 1. Build the Docker Image

```bash

```

### 2. Run

Insert the run command for the relevant framework from the table above in the part that says `<run command>`. Running the container will do the following: 1. download the MNIST dataset (it will be deleted when the container finishes), 2. train the model, 3. test the model, 4. print a summary of the results.

```bash

```


*We also provide a VS Code 'devcontainer' specification. If you use VS Code (with the remote development extension pack), you can setup a development environment by cloning the repo and running `Remote Containers: Open folder in Container` from the command palette*.
