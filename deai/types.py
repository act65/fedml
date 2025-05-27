from typing import Any
from flax import struct

@struct.dataclass
class Node:
    id: int
    ip: str
    port: int
    type: str

@struct.dataclass
class TrainState:
    params: Any
    opt_state: Any

@struct.dataclass
class Message:
    var: Any  # JAX pytree
    metadata: dict   # sender_id, dataset_size, timestamp, etc.

@struct.dataclass
class WorkerState():
    train_state: TrainState
    extras: dict