<div align="center">

# MNIST in Every Framework

</div>

## Summary

*I hire not the researcher who has trained 10000 models once, I hire the researcher who has trained 1 model 10000 times* - Bruce Lee, probably

We implement and train a MNIST MLP classifier in increasingly unorthodox
frameworks and languages. The goal is to practice coding in different styles and
to learn new languages. Are you a deep learning researcher who wants to learn a
new language? Has PhD life turned your brain into mush that can only understand
28*28 pixel handwritten characters? You've come to the right place!

|                   | Implemented?                | Run command:                | Data: | Training: | Network: | Autograd:
|-------------------|-----------------------------|--------------------------   |-      |-          |-         |-           |
| PyTorch           | <ul><li>- [x] </li></ul>    | `python pytorch/main.py`    | torchvision | Custom | torch.nn | torch |
| PyTorch-Lightning | <ul><li>- [x] </li></ul>    | `python lightning/main.py`  | torchvision | pl.trainer | torch.nn| torch |
| TensorFlow        | <ul><li>- [x] </li></ul>    | `python tensorflow/main.py` | tfds | Keras | Keras | tf |
| JAX               | <ul><li>- [x] </li></ul>    | `python haiku/main.py`      | tfds | Custom | haiku | JAX |
| NumPy             | <ul><li>- [ ] </li></ul>    | `python numpy/main.py`      | Custom | Custom | Custom | Custom |
| Julia             | <ul><li>- [ ] </li></ul>    | `julia julia/main.jl`       | Custom | Custom | Flux | Zygote |
| Rust              | <ul><li>- [ ] </li></ul>    | `cargo run`                 | Custom | ? | ? | ? |
| Haskell           | <ul><li>- [ ] </li></ul>    | `N/A`                       | | | | |
| OCaml             | <ul><li>- [ ] </li></ul>    | `N/A`                       | | | | |
| C                 | <ul><li>- [ ] </li></ul>    | `N/A`                       | | | | |
| Assembly?!        | <ul><li>- [ ] </li></ul>    | `N/A`                       | | | | |

The table above shows the progress of each project. Making an MNIST classifier
generally requires some code to do at least four things: load and read the
files, define and manage network architecture and parameters, make a training
loop and training steps, and compute gradients. I show roughly what each project
does for each of these tasks in the table. You will notice that I have to do more
custom stuff as I move down the list towards languages that are less popular for
scientific computing and ML.


## Installation and Running

Since we use lots of different languages and frameworks, this project would be a
nightmare to install from scratch. Instead, we provide a Dockerfile for
installing and running the project with minimal pain. The only dependency you need
is a working Docker installation.

### Prerequisites

Clone this repo:

```bash
git clone https://github.com/Charl-AI/MNIST-in-every-framework.git
cd MNIST-in-every-framework
```

Check that your Docker installation is working:

```bash
docker run hello-world
```

### Option 1: Build and run the Docker image manually

First, build the image:

```bash
docker build -f .devcontainer/Dockerfile . -t mnist-in-every-framework
```

Next, run the desired project, replacing `<run command>` with the command to run for the project. The MNIST dataset will be downloaded automatically upon running the container. Outputs of the project will be logged to your local filesystem in the `outputs/` directory so you can view the results.

```bash
docker run --rm -it -v $(pwd)/outputs:/app/outputs mnist-in-every-framework <run command>
```

### Option 2: Use VSCode Remote Containers

If you use VSCode with the remote containers extension, you can install the dependencies and open a VSCode development environment for this project by simply running `Remote Containers: Open folder in Container` from the command palette. You can then run any project you want by pasting the run command into the integrated terminal.

### Option 3: Codespaces!

(New as of Nov 2022: GitHub provides a free tier with 80 hours per month of compute.)

You can open this repo **without any local installation** in a cloud-hosted 'codespace'. The dependencies will be automatically installed on the remote cloud machine, which you can access through your browser, VSCode, or even PyCharm. Simply start the codespace using the button in the GitHub UI. This is the easiest way to try out this project.
