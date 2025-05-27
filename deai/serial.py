import jax
import base64
import pickle
import jax.numpy as jnp
from jax import tree

class PyTreeSerializer:
    @staticmethod
    def serialize(pytree):
        # Convert PyTree to structure + leaves
        leaves, tree_def = tree.flatten(pytree)

        # Serialize both components
        serialized = {
            "tree_def": base64.b64encode(pickle.dumps(tree_def)).decode("utf-8"),
            "leaves": [x.tolist() if isinstance(x, jax.Array) else x for x in leaves]
        }
        return serialized

    @staticmethod
    def deserialize(data):
        tree_def = pickle.loads(base64.b64decode(data["tree_def"]))
        leaves = [jnp.array(x) for x in data["leaves"]]
        return tree.unflatten(tree_def, leaves)