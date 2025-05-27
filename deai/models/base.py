import jax.numpy as jnp
from jax import value_and_grad, vmap
from jax.nn import initializers

from inspect import signature
from functools import partial

import flax.linen as nn

class MLP(nn.Module):
    depth: int
    width: int
    n_out: int
    act: callable = nn.swish
    
    @nn.compact
    def __call__(self, x, *args, **kwargs):
        s = x.shape
        x = jnp.reshape(x, -1)

        for _ in range(self.depth-1):
            x = nn.Dense(
                features=self.width,
                kernel_init=initializers.orthogonal(),
                )(x)
            x = self.act(x)
        x = nn.Dense(features=self.n_out)(x)
        return x

def batch_avg_wrapper(loss_fn):
    # assumes loss_fn takes (params, *args)
    n = len(signature(loss_fn).parameters)
    b_loss = vmap(loss_fn, in_axes=(None, ) + (n-1) * (0,))
    def batch_loss(*args):
        return jnp.mean(b_loss(*args))
    return batch_loss

class Model():
    def __init__(self, network, observation_spec, **kwargs):
        self.network = network
        self.observation_spec = observation_spec

        self.b_loss_fn = batch_avg_wrapper(self.loss_fn)
        self.b_eval_fn = batch_avg_wrapper(self.eval_fn)
            
        self.dldparams = value_and_grad(self.b_loss_fn)

    def loss_fn(self, params, x, t):
        # simple regression loss
        y = self.network.apply(params, x)
        assert y.shape == t.shape
        return jnp.sum(jnp.square(y - t))
    
    def eval_fn(self, params, x, t):
        # just use MSE for now
        return self.loss_fn(params, x, t)

    def initial_params(self, key):
        sample_input = self.observation_spec()
        return self.network.init(key, sample_input)
