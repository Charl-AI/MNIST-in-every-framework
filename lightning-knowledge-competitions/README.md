# lightning-knowledge-competitions

Solutions to the Kaggle Knowledge Competitions in [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/).
Pytorch lightning is primarily aimed at researchers looking for the flexibility and comfortable API of Pytorch with a few
neat features for organising the research code and abstracting away engineering code.

The code looks similar to the vanilla Pytorch code, but notice how we don't need to write our own training loops. The logic of each training step is contained in the model 'LightningModule' which we pass to a trainer which is part of the library. The 'LightningDataModule' serves a similar purpose - it allows us to contain the logic for splitting and loading our dataset in one place. Also, the pre-made trainer has some features to save us time such as moving tensors between CPU and GPU automatically, mixed precision training, and some basic logging functionality.
