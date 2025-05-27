from jax import random
import jax.numpy as jnp

from deai.data import FeSimple

import matplotlib.pyplot as plt

def test():

    dims = 4
    dataset = FeSimple(silo_id=0, n_mixtures=3, dims=dims)

    key = random.PRNGKey(0)
    batch_size = 8
    data_generator = dataset.all_silo_data_generator(key, batch_size)

    x, y = next(data_generator)

    print(x.shape, y.shape)
    assert x.shape == (batch_size, dims)
    assert y.shape == (batch_size, )
    assert x.dtype == jnp.float32
    assert y.dtype == jnp.float32

    x, y = next(data_generator)

if __name__ == "__main__":
    test()
