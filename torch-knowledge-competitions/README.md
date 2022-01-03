# torch-knowledge-competitions

Solutions to the Kaggle Knowledge Competitions in [Pytorch](https://pytorch.org/).
Pytorch is a popular framework for deep learning and is particularly widespread in research. It has a stable API and good documentation which makes it great for developers.

In Pytorch, we typically need to define three things: our model, our dataset, and our training loop. These are contained in the `models/`, `datasets/`, and `trainer/` directories respectively. The flexibility of Pytorch is one of its best features, however, it sometimes leaves you writing somewhat repetitive code (for example, most training loops will look pretty similar). In this project, we use notebooks instead of a `main.py` file for ease of visualisation. In larger projects, it is a good idea to not use notebooks and to create a CLI or use config files/[Hydra](https://hydra.cc/docs/intro/) to specify configurations.
