import pytest
from jax import random
import jax.numpy as jnp

import contextlib
from deai.data import GroupedFeMNIST

import matplotlib.pyplot as plt

@pytest.mark.parametrize("train_mode, N, batch_size, expect_error", [
    (True, 6, 8, None), 
    (False, 3, 7, None),
    (True, 1000, 12, ValueError),
    ])
def test_gfmnist(train_mode, N, batch_size, expect_error):
    datadir = '/home/act65/Documents/work/sparsity/leaf/data/femnist/data'
    expected_context = pytest.raises(expect_error) if expect_error else contextlib.nullcontext()

    with expected_context:
        dataset = GroupedFeMNIST(silo_id=1, datadir=datadir, n_silos=N, train=train_mode)

        key = random.PRNGKey(0)
        data_generator = dataset.infinite_data_generator(key, batch_size)

        x, y = next(data_generator)

        assert x.shape == (batch_size, 28, 28, 1)
        assert y.shape == (batch_size,)
        assert x.dtype == jnp.float32
        assert y.dtype == jnp.int32

def plot_fmnist(train=True, N=6, batch_size=8):
    key = random.PRNGKey(0)
    counter = 0
    for k in range(N):
        datadir = '/home/act65/Documents/work/sparsity/leaf/data/femnist/data/'
        dataset = GroupedFeMNIST(silo_id=k, datadir=datadir, n_silos=N, train=train)
        data_generator = dataset.infinite_data_generator(key, batch_size)
        x, y = next(data_generator)

        for i in range(batch_size):
            plt.subplot(N, batch_size, counter+1)
            plt.imshow(x[i, :, :, 0], cmap='gray')
            plt.axis('off')
            plt.title(f"{y[i]}")

            counter += 1

    plt.show()

def dataset_size(train=False, N=2):
    datadir = '/home/act65/Documents/work/sparsity/leaf/data/femnist/data/'
    key = random.PRNGKey(0)
    batch_size = 8

    for i in range(N):
        dataset = GroupedFeMNIST(silo_id=i, datadir=datadir, n_silos=N, train=train)
        counter = 0

        data_generator = dataset.data_generator(key, batch_size)
        for x, y in data_generator:
            counter += len(x)
        print(f"Dataset {i} has {counter} samples")


if __name__ == "__main__":
    # plot_fmnist()
    dataset_size()