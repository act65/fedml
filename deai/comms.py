import threading
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify
import requests

import logging
logging.getLogger('werkzeug').setLevel(logging.WARNING)

from deai.logging import FileLogger
from deai.serial import PyTreeSerializer
from deai.types import Message

def parse_ips(ips):
    if isinstance(ips, str):
        return ips.split(',')
    else:
        return ips
    
def parse_ports(ports):
    if isinstance(ports, str):
        return [int(port) for port in ports.split(',')]
    else:
        return ports

class CommunicationManager:
    """
    Handles all peer-to-peer communication concerns.
    This communication uses a simple REST API to send and receive messages.
    """
    def __init__(self, ip, port, peer_ips, peer_ports):
        self.app = Flask(__name__)
        self.ip = ip
        self.port = port
        self.peer_ips = peer_ips
        self.peer_ports = peer_ports
        self.update_lock = threading.Lock()  # v RLock?
        # self.executor = ThreadPoolExecutor(max_workers=4)
        self.message_queue = []
        self.serializer = PyTreeSerializer()
        self.logger = FileLogger(f"logs/comms-{self.ip}-{self.port}")
        
        self.app.add_url_rule('/push', 'receive', 
                            self.receive, methods=['POST'])

        # Start server in background thread
        self.server_thread = threading.Thread(target=self.run_server)
        self.server_thread.start()

    def run_server(self):
        self.app.run(host='0.0.0.0', port=self.port)

    def send(self, peer_ip, peer_port, message):
        assert isinstance(message, Message)
        # self.executor.submit(
        requests.post(f'http://{peer_ip}:{peer_port}/push',
                json={'message': self.serializer.serialize(message)},
                timeout=20.0)  # Fire-and-forget
        # )

    def receive(self):
        if not request.json or 'message' not in request.json:
                return jsonify({'status': 'ERROR', 'reason': 'No message'}), 400
        
        serialized_message = request.json['message']

        try:
            message = self.serializer.deserialize(serialized_message)
        except Exception as e:
            self.logger.error(f"Deserialization error: {e}")
            return jsonify({'status': 'ERROR', 'reason': 'Invalid message format'}), 400

        with self.update_lock:
            self.message_queue.append(message)  # TODO this buffer should filter out old messages?! duplicates?
        self.logger.info(f"Message received and added to queue. Queue size: {len(self.message_queue)}")
        return jsonify({'status': 'ACK'})

    def get_available_messages(self):
        # Returns all received messages since last check
        messages = self.message_queue
        self.message_queue = []
        return messages
    
    @property
    def n_available_messages(self):
        return len(self.message_queue)
    
    def shutdown(self):
        """
        Shuts down the server thread and the thread pool executor.
        """
        self.logger.info("Shutting down CommunicationManager...")
        self._executor.shutdown(wait=True)  # Wait for all send tasks to complete
        # No explicit shutdown for server_thread as it's a daemon thread and will exit with the main thread
        self.logger.info("CommunicationManager shutdown complete.")


from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class CommunicationPolicy(ABC):
    """Base class for communication scheduling policies"""
    @abstractmethod
    def should_send(self, step: int, state: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def should_receive(self, step: int, state: Dict[str, Any]) -> bool:
        pass

class PeriodicPolicy(CommunicationPolicy):
    """Send periodically, receive whenever messages are available"""
    def __init__(self, send_interval: int = 1, receive_interval: int = 1):
        self.send_interval = send_interval
        self.receive_interval = receive_interval
        
    def should_send(self, step: int, state: Dict[str, Any]=None) -> bool:
        return (step % self.send_interval == 0) and (step > 0) # Skip first step
    
    def should_receive(self, step: int, state: Dict[str, Any]=None) -> bool:
        return (step % self.receive_interval == 0) and (state['n_available_messages'] > 0)

class SyncPolicy(CommunicationPolicy):
    """Send and receive synchronously"""
    def __init__(self, n_peers: int):
        self.n_peers = n_peers
        
    def should_send(self, step: int, state: Dict[str, Any]=None) -> bool:
        return True  # Always send
    
    def should_receive(self, step: int, state: Dict[str, Any]=None) -> bool:
        if state['n_available_messages'] >= self.n_peers:
            return True
        else:
            return False

# class PeroidicSend_SyncRecieve(PeriodicPolicy, SyncPolicy):
#     """Combines PeriodicPolicy's send and SyncPolicy's receive"""
#     def __init__(self, send_interval: int, n_peers: int):
#         # Explicitly initialize both parent classes
#         PeriodicPolicy.__init__(self, send_interval)
#         SyncPolicy.__init__(self, n_peers)

#     def should_send(self, step: int, state: Dict[str, Any] = None) -> bool:
#         return PeriodicPolicy.should_send(self, step, state)
    
#     def should_receive(self, step: int, state: Dict[str, Any] = None) -> bool:
#         return SyncPolicy.should_receive(self, step, state)

# NOTE if we make should_send variable (aka if we use a different num of steps in the inner train loop)
# then we need to worry about balancing the gradients