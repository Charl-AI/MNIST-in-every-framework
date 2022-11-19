<div align="center">

# MNIST in Every Framework

</div>

## Summary

*I hire not the researcher who has trained 10000 models once, I hire the researcher who has trained 1 model 10000 times* - Bruce Lee, probably

We implement and train a MNIST MLP classifier in increasingly unorthodox frameworks and languages. The goal is to practice coding in different styles. Are you a deep learning researcher who wants to learn a new language, but hates web development?? You've come to the right place!

|                   | Implemented? | Run command: |
|-------------------|-----------------------------|--------------------------|
| PyTorch           | <ul><li>- [x] </li></ul>    | `bash pytorch/run.sh` |
| PyTorch-Lightning | <ul><li>- [x] </li></ul>    | `bash pytorch-lightning/run.sh` |
| TensorFlow        | <ul><li>- [x] </li></ul>    | `bash tensorflow/run.sh` |
| JAX + Haiku       | <ul><li>- [x] </li></ul>    | `bash jax/run.sh` |
| Julia             | <ul><li>- [ ] </li></ul>    | `bash julia/run.sh` |
| Rust              | <ul><li>- [ ] </li></ul>    | `bash rust/run.sh` |
| Haskell           | <ul><li>- [ ] </li></ul>    | `bash haskell/run.sh` |
| OCaml             | <ul><li>- [ ] </li></ul>    | `bash ocaml/run.sh` |
| C                | <ul><li>- [ ] </li></ul>    | `bash c/run.sh` |
| Assembly?!        | <ul><li>- [ ] </li></ul>    | `bash assembly/run.sh` |


## Installation and Running

We provide a Dockerfile for installation and running the project. The image requires your host machine to have a GPU + driver capable of running CUDA 11.3 (you can check this by running `nvidia-smi` and looking at the 'CUDA Version' field). If you don't, you will have to install everything manually.

### 0. Prerequisites

Clone this repo. Next, install Docker with the NVIDIA container runtime. Instructions can be found [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

### 1. Build the Docker Image

```bash

```

### 2. (Optional) Check to make sure GPU is working

```bash
docker run --gpus=all ... nvidia-smi
```

### 3. Run

Insert the run command for the relevant framework from the table above in the part that says `<run command>`. Running the container will do the following: 1. download the MNIST dataset (it will be deleted when the container finishes), 2. train the model, 3. test the model, 4. print a summary of the results.

```bash

```


*We also provide a VS Code 'devcontainer' specification. If you use VS Code (with the remote development extension pack), you can setup a development environment by cloning the repo and running `Remote Containers: Open folder in Container` from the command palette.*
