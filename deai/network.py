from deai.types import Node
from deai.worker import FederatedWorker, FederatedServer, FederatedEvaluator

class Network:
    """
    Contains all the info about nodes, their ips, ports, ... and their connections.
    """
    def __init__(self, nodes: dict, connections: dict):
        self.nodes = nodes
        self.connections = connections

    def __str__(self):
        return f"Nodes: {self.nodes}, Connections: {self.connections}"

    def __repr__(self):
        return self.__str__()

    def get_node(self, node_id: int) -> Node:
        return self.nodes[node_id]

    def get_connections(self, node_id: int) -> list[int]:
        return self.connections[node_id]

    def get_peer_nodes(self, node_id: int) -> list[Node]:
        peer_ids = self.get_connections(node_id)
        return [self.get_node(node_id) for node_id in peer_ids]

    def get_peer_ips(self, node_id: int) -> list[str]:
        peer_nodes = self.get_peer_nodes(node_id)
        return [node.ip for node in peer_nodes]

    def get_peer_ports(self, node_id: int) -> list[int]:
        peer_nodes = self.get_peer_nodes(node_id)
        return [node.port for node in peer_nodes]

    def get_node_ip(self, node_id: int) -> str:
        return self.get_node(node_id).ip

    def get_node_port(self, node_id: int) -> int:
        return self.get_node(node_id).port

    def get_node_type(self, node_id: int) -> str:
        return self.get_node(node_id).type

def construct_centralized(ips: list[str], ports: list[int]) -> Network:
    """
    Constructs a centralized network topology.

    By convention:
    - Node 0: FederatedServer
    - Node 1: FederatedEvaluator
    - Nodes 2 onwards: FederatedWorker

    Args:
        ips (list[str]): List of IP addresses for the nodes.
        ports (list[int]): List of port numbers for the nodes.

    Returns:
        Network: A Network object representing the centralized topology.
    """
    nodes = dict()
    # add server node
    nodes[0] = Node(id=0, ip=ips[0], port=ports[0], type='FederatedServer')
    # add eval node
    nodes[1] = Node(id=1, ip=ips[1], port=ports[1], type='FederatedEvaluator')
    # add worker nodes
    for i in range(2, len(ips)):
        nodes[i] = Node(id=i, ip=ips[i], port=ports[i], type='FederatedWorker')

    connections = dict()
    # server node is connected to all.
    connections[0] = list(range(1, len(ips)))
    # eval node is connected to server node.
    connections[1] = [0]
    # worker nodes are connected to server node.
    for i in range(2, len(ips)):
        connections[i] = [0]

    return Network(nodes, connections)