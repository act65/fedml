import jax.numpy as jnp
import flax.linen as nn
from jax.nn import log_softmax

from deai.models.base import Model

class CNN(nn.Module):
    n_out: int = 10
    depth: int = 2
    width: int = 256
    act: callable = nn.relu
    kernel_size: tuple = (3, 3)

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        for _ in range(self.depth):
            x = nn.Conv(features=self.width, kernel_size=self.kernel_size, padding='SAME')(x)
            x = nn.relu(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape(-1)
        x = nn.Dense(features=self.n_out)(x)
        return x
        
class Categorical(Model):
    def __init__(self, network, observation_spec, n_classes, **kwargs):
        super().__init__(network, observation_spec, **kwargs)
        self.n_classes = n_classes

    def loss_fn(self, params, x, t):
        # TODO use sparse CCE
        t = jnp.eye(self.n_classes)[t]
        y = log_softmax(self.network.apply(params, x))
        assert t.size == y.size
        return -jnp.sum(y * t)
    
    def eval_fn(self, params, x, t):
        # return accuracy
        y = self.network.apply(params, x)
        assert t.size == 1
        p = jnp.argmax(y, axis=-1).flatten().astype(jnp.int32)
        t = t.flatten().astype(jnp.int32)
        return jnp.equal(p, t).astype(jnp.float32)