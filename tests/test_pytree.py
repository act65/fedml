import jax.numpy as jnp
from jax import tree

from deai.utils import serialize_pytree, deserialize_pytree

def test_serialize_deserialize_pytree():
    pytree = {
        'a': jnp.array([1, 2, 3]),
        'b': jnp.array([4, 5, 6]),
        'c': {
            'd': jnp.array([7, 8, 9])
        }
    }
    serialized = serialize_pytree(pytree)
    deserialized = deserialize_pytree(serialized)

    for a, b in zip(tree.leaves(pytree), tree.leaves(deserialized)):
        assert jnp.all(a == b)

    assert tree.structure(pytree) == tree.structure(deserialized)


if __name__ == "__main__":
    test_serialize_deserialize_pytree()