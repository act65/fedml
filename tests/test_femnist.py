from jax import random
import jax.numpy as jnp

from deai.data import FeMNIST

import matplotlib.pyplot as plt

def test_fmnist():

    datadir = '/home/act65/Documents/work/sparsity/leaf/data/femnist/data'
    dataset = FeMNIST(silo_id=1, datadir=datadir)

    key = random.PRNGKey(0)
    data_generator = dataset.infinite_data_generator(key, 4)

    x, y = next(data_generator)

    assert x.shape == (4, 28, 28, 1)
    assert y.shape == (4,)
    assert x.dtype == jnp.float32
    assert y.dtype == jnp.int32

    x, y = next(data_generator)

def plot_fmnist():
    key = random.PRNGKey(0)
    N = 6
    batch_size = 8
    counter = 0
    for k in range(N):
        datadir = '/home/act65/Documents/work/sparsity/leaf/data/femnist/data/train'
        dataset = FeMNIST(silo_id=k, datadir=datadir)
        data_generator = dataset.infinite_data_generator(key, batch_size)
        x, y = next(data_generator)

        for i in range(batch_size):
            plt.subplot(N, batch_size, counter+1)
            plt.imshow(x[i, :, :, 0], cmap='gray')
            plt.axis('off')
            plt.title(f"{y[i]}")

            counter += 1

    plt.show()


if __name__ == "__main__":
    test_fmnist()
    # plot_fmnist()