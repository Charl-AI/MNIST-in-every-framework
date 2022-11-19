import haiku as hk


@hk.without_apply_rng
@hk.transform
def mnist_mlp(x):
    model = hk.nets.MLP([512, 512, 10])
    return model(x)
