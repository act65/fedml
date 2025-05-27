from jax import random, tree
from deai.trainer import Trainer
import jax.numpy as jnp

from deai.data import SimpleDataset, FeSimple, FeMNIST, GroupedFeMNIST    
from deai.models.base import MLP, Model
from deai.models.mnist import CNN, Categorical
from deai.comms import PeriodicPolicy, SyncPolicy
from deai.aggregators import Average, Latest, WeightedAverage
from deai.algorithms import FedAvg

def build_trainer(dataset, model_config, trainer_config):
    """
    A utility function to build a trainer from a config.
    """

    # Initialize network
    network_config = model_config.network
    if network_config.name == "MLP":
        net = MLP(network_config.depth, network_config.width, dataset.shapes[1])
    elif network_config.name == "CNN":
        net = CNN(n_out=dataset.n_classes, depth=network_config.depth, width=network_config.width)
    else:
        raise ValueError("Unknown network name")
    
    # obs_spec = lambda : tree.map(lambda s: jnp.ones(s), dataset.shapes[0])
    obs_spec = lambda : jnp.ones(dataset.shapes[0])
    # Initialize model
    if model_config.loss == "regression":
        model = Model(net, obs_spec)
    elif model_config.loss == "categorical":
        model = Categorical(net, obs_spec, dataset.n_classes)

    # Initialize trainer
    return Trainer(model, **trainer_config)

def build_dataset(config, **kwargs):
    if config.name == "simple":
        return SimpleDataset(**config, **kwargs)
    elif config.name == "femnist":
        return FeMNIST(**config, **kwargs)
    elif config.name == "gfemnist":
        return GroupedFeMNIST(**config, **kwargs)
    elif config.name == "fesimple":
        return FeSimple(**config, **kwargs)
    else:
        raise ValueError("Unknown dataset name")
    
def build_comms_policy(policy_config, network_config):
    n_peers = len(network_config.peer_ips)  # BUG this assumes all peers are workers
    if policy_config.name == "periodic":
        return PeriodicPolicy(**policy_config.args)
    elif policy_config.name == "sync":
        return SyncPolicy(n_peers=n_peers-1)
    else:
        raise ValueError("Unknown comms policy name")
    
def build_aggregator(aggregator_name):
    if aggregator_name == "average":
        return Average()
    elif aggregator_name == "weighted_average":
        return WeightedAverage()  # TODO how to get weights here!?
    elif aggregator_name == "latest":
        return Latest()
    else:
        raise ValueError("Unknown aggregator name")

def build_algo(algo_config):
    if algo_config.name == "fedavg":
        server_agg = build_aggregator(algo_config.server_aggregator)
        worker_agg = build_aggregator(algo_config.worker_aggregator)
        return FedAvg(server_agg, worker_agg)
    else:
        raise ValueError("Unknown algorithm name")

class PrivacyManager:
    def __init__(self, seed, sigma, clip_val):
        self.key = random.PRNGKey(seed)
        self.sigma = sigma
        self.clip_val = clip_val

    def __call__(self, var):
        noisy_grad = self.add_noise(var, self.sigma)
        return self.clip(noisy_grad, self.clip_val)
        
    def add_noise(self, var, sigma):
        self.key, subkey = random.split(self.key)
        return add_noise(subkey, var, sigma)
    
    def clip(self, var, clip):
        return tree.map(lambda x: jnp.clip(x, -clip, clip), var)

def add_noise(key, grads, noise_scale):
    leaves, treedef = tree.flatten(grads)
    # Add noise to the gradient leaf (x + noise)
    add_noise_to_leaf = lambda x, subkey: x + random.normal(subkey, x.shape) * noise_scale
    keys = random.split(key, len(leaves))
    noisy_leaves = map(add_noise_to_leaf, leaves, keys)
    noisy_tree = tree.unflatten(treedef, noisy_leaves)
    return noisy_tree

"""
A set of utils for converting lists of http addresses to a network of peers.
"""

def ring_topology(urls):
    """
    Create a ring topology from a list of URLs.
    """
    return {url: [urls[(i+1) % len(urls)]]
                  for i, url in enumerate(urls)}

def all_to_all_topology(urls):
    """
    Create an all-to-all topology from a list of URLs.
    """
    return {url: [url2 for url2 in urls if url2 != url]
                  for url in urls}

def centralized_topology(ips, ports):
    """
    Create a centralized topology from a list of URLs.
    """
    mapping = dict()
    # workers
    for i in range(0, len(ips)-1):
        mapping[(ips[i], ports[i])] = [(ips[-1], ports[-1])]
    # central server
    mapping[(ips[-1], ports[-1])] = [(ips[i], ports[i]) for i in range(0, len(ips)-1)]
    return mapping
