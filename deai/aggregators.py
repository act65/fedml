import jax.numpy as jnp
from jax import tree, jit

from deai.comms import Message

from operator import itemgetter

class Aggregator:
    # aggregator: function that takes a local message and a list of peer messages
    # and returns the gradient update to be applied to the model
    def __call__(self, messages: list[Message]):
        if len(messages) > 0:
            return self.aggregate(messages)
        else:
            raise ValueError("No messages to aggregate")
        
    def aggregate(self, messages: list[Message]) -> jnp.ndarray:
        raise NotImplementedError
        
class Average(Aggregator):
    def aggregate(self, messages: list[Message]):
        vars = [m.var for m in messages]
        return tree.map(lambda *x: jnp.mean(jnp.stack(x), axis=0), *vars)
        
class Latest(Aggregator):
    def aggregate(self, messages: list[Message]):
        latest = max(messages, key=lambda x: x.metadata['timestamp'])
        return latest.var

class WeightedAverage(Aggregator):
    def __init__(self, weights: dict):
        # needs to know the weights of each worker at init
        self.weights = weights
        self.total = sum(weights.values())
        # TODO would be nice to jit this?

    # NOTE in the unreliable network case, these weighted averages could be missing some messages
    # thus the weights would be incorrect


    def aggregate(self, messages: list[Message]):
        # BUG silo_id != worker_id
        weights = [self.weights[int(m.metadata['worker_id'])] for m in messages]

        # Weighted average calculation
        def weighted_mean(*arrays):
            return sum(w * arr for w, arr in zip(weights, arrays)) / self.total
            
        return tree.map(weighted_mean, *[m.var for m in messages])
    
# class CumlativeWeightedAverage(WeightedAverage):
#     """

    
#     """
#     def __init__(self, decay_rate=0.9):
#         self.decay_rate = decay_rate
#         self.weights = dict()

#     def aggregate(self, messages: list[Message]):


class Filter:
    # filter: function that takes a list of messages and returns a subset of them
    def __call__(self, messages: list[Message]) -> list[Message]:
        assert len(messages) > 0
        return self.filter(messages)
    
    def filter(self, messages: list[Message]) -> list[Message]:
        raise NotImplementedError
    
class LatestPerWorker(Filter):
    def filter(self, messages: list[Message]) -> list[Message]:
        latest_per_worker = {}
        for m in messages:
            worker_id = int(m.metadata['worker_id'])
            if worker_id not in latest_per_worker:
                latest_per_worker[worker_id] = m
            else:
                if m.metadata['timestamp'] > latest_per_worker[worker_id].metadata['timestamp']:
                    latest_per_worker[worker_id] = m
        return list(latest_per_worker.values())

# NOTE others could be timing sensitive?
# class AsyncBuffer:
#     """Stores updates with metadata for freshness weighting"""
#     def __init__(self, max_staleness=5):
#         self.updates = []
#         self.max_staleness = max_staleness
        
#     def add_update(self, message: FederatedMessage):
#         self.updates.append(message)
#         # Implement staleness-based pruning
#         self.updates = [m for m in self.updates 
#                        if (current_round - m.metadata['round']) <= self.max_staleness]
