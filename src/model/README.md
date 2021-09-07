# Models and Lightning modules are here

Lightning modules define the entire *system* including the model, training step etc. Lightning Bolts, ```pip install lightning-bolts``` , includes some pre-made models which may be of use.

Store the building blocks of models in the components subdirectory.

A typical lightning module looks like this, more info at https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html:

```python
class LitModel(pl.LightningModule):
    def __init__(self, layer_1_dim):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = nn.Linear(28 * 28, layer_1_dim)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitModel")
        parser.add_argument("-l, --layer_1_dim", help="first layer dimensions", type=int, default=12)
        return parent_parser
