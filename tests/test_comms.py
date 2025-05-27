import pytest
from unittest import mock
import jax.numpy as jnp

from deai.types import Message
from deai.comms import CommunicationManager

import requests
from concurrent.futures import ThreadPoolExecutor

@pytest.fixture
def mock_thread():
    with mock.patch('threading.Thread') as mock_thread:
        yield mock_thread

@pytest.fixture
def message():
    return Message(
        var=jnp.ones((10, 10)),
        metadata={'worker_id': 0, 'timestamp': 0}
    )

class TestCommunicationManager:  # Group tests in a class for better organization

    def test_receive_message(self, mock_thread, message): # Renamed self for clarity
        # Arrange - Setup phase
        cm = CommunicationManager(ip='127.0.0.1', port=5000, peer_ips=[], peer_ports=[])
        serialized = cm.serializer.serialize(message)
        client = cm.app.test_client()

        # Act - Execution phase
        response = client.post('/push', json={'message': serialized})

        # Assert - Verification phase
        assert response.status_code == 200
        assert response.json == {'status': 'ACK'}
        assert len(cm.message_queue) == 1
        assert jnp.array_equal(cm.message_queue[0].var, message.var) # Use jnp.array_equal for jax arrays
        assert cm.message_queue[0].metadata == message.metadata

    def test_send_message(self, mock_thread, message):
        # Arrange
        with mock.patch('requests.post') as mock_post:
            sender = CommunicationManager(ip='127.0.0.1', port=5000, peer_ips=[], peer_ports=[])
            peer_ip, peer_port = '192.168.1.1', 6000

            # Act
            sender.send(peer_ip, peer_port, message)

            # Assert
            expected_url = f'http://{peer_ip}:{peer_port}/push'
            expected_json = {'message': sender.serializer.serialize(message)}
            mock_post.assert_called_once_with(expected_url, json=expected_json, timeout=20.0)
            assert mock_post.call_count == 1

    def test_get_available_messages(self, mock_thread, message):
        # Arrange
        cm = CommunicationManager(ip='127.0.0.1', port=5000, peer_ips=[], peer_ports=[])
        msg1, msg2 = message, message  # Consider creating distinct messages if needed for future tests
        cm.message_queue = [msg1, msg2]

        # Act
        messages = cm.get_available_messages()

        # Assert
        assert messages == [msg1, msg2]
        assert cm.message_queue == [] # Queue should be empty after getting messages

    def test_n_available_messages(self, mock_thread, message):
        # Arrange
        cm = CommunicationManager(ip='127.0.0.1', port=5000, peer_ips=[], peer_ports=[])
        cm.message_queue = [message, message]

        # Act & Assert (combined for simple property check)
        assert cm.n_available_messages == 2

    # def test_concurrent_receives(self, message): # Removed mock_thread fixture
    #     cm = CommunicationManager(ip='127.0.0.1', port=5000, peer_ips=[], peer_ports=[])
    #     num_messages = 3
    #     test_messages = [message for _ in range(num_messages)]
    #     serialized_messages = [cm.serializer.serialize(msg) for msg in test_messages]
    #     client = cm.app.test_client()

    #     def post_message(serialized):
    #         print(f"Thread starting to post message...")
    #         try:
    #             response = client.post('/push', json={'message': serialized}, timeout=5)
    #             print(f"Thread received response. Status: {response.status_code}")
    #             assert response.status_code == 200
    #         except requests.exceptions.Timeout:
    #             print("Thread timed out during client.post")
    #             pytest.fail("Request timed out")
    #         except Exception as e:
    #             print(f"Thread encountered error during client.post: {e}")
    #             pytest.fail(f"Error during request: {e}")

    #     with ThreadPoolExecutor(max_workers=2) as executor:
    #         futures = [executor.submit(post_message, sm) for sm in serialized_messages]
    #         for future in futures:
    #             future.result()

    #     print("All threads completed.")

    #     assert len(cm.message_queue) == num_messages
    #     received_message_vars = [msg.var for msg in cm.message_queue]
    #     test_message_vars = [msg.var for msg in test_messages]
    #     received_message_vars.sort(key=lambda x: x.sum())
    #     test_message_vars.sort(key=lambda x: x.sum())
    #     for received_var, expected_var in zip(received_message_vars, test_message_vars):
    #         assert jnp.array_equal(received_var, expected_var)
    #     received_message_metadatas = [msg.metadata for msg in cm.message_queue]
    #     test_message_metadatas = [msg.metadata for msg in test_messages]
    #     assert received_message_metadatas == test_message_metadatas
