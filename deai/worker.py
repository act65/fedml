
import jax
import jax.numpy as jnp
from jax import random, tree
import datetime

from deai.types import Message, TrainState, WorkerState
from deai.utils import build_trainer, build_dataset, PrivacyManager, build_comms_policy, build_algo
from deai.comms import CommunicationManager
from deai.logging import FlexibleLogger, PrintLogger, FileLogger

from omegaconf import DictConfig

class BaseWorker:
    def __init__(self, worker_id, comms_config):
        self.worker_id = worker_id
        self.logger = PrintLogger(f"Worker {worker_id}")
        # self.logger = FileLogger(f"logs/worker-{worker_id}")
        self.comms_manager = CommunicationManager(**comms_config)

    def create_message(self, var):
        return Message(
            var=var,
            metadata={
                'worker_id': self.worker_id,
                'timestamp': datetime.datetime.now().timestamp(),
                'port': self.comms_manager.port
            }
        )

    def broadcast(self, var):
        # Push to peers TODO non-blocking
        for peer_ip, peer_port in zip(self.comms_manager.peer_ips, self.comms_manager.peer_ports):
            message = self.create_message(var)  # most cases, this will be the same for all peers
            self.comms_manager.send(peer_ip, peer_port, message)

class FederatedWorker(BaseWorker):
    """
    This class handles working in a centralized setting.
    """
    def __init__(self, worker_id, comms_config, dataset_config, model_config, trainer_config, dp_config, algo_config):
        super().__init__(worker_id, comms_config)
        self.logger.info(f"Args: {worker_id}, {comms_config}, {dataset_config}, {model_config}, {trainer_config}, {dp_config}, {algo_config}")

        self.dp = PrivacyManager(**dp_config)

        self.fl_algo = build_algo(algo_config.algorithm)
        self.dataset = build_dataset(dataset_config, train=True)
        self.comms_policy = build_comms_policy(algo_config.worker_comms_policy, comms_config)
        self.trainer = build_trainer(self.dataset, model_config, trainer_config)

    def create_message(self, var):
        # overwrite create_message to add noise
        # since we want to keep the data private
        # NOTE if you add noise seperately for each peer this introduces a secutiry vulnerability
        # since a malicious actor could be running multiple peers and average out the noise
        var = self.dp(var)
        message = super().create_message(var)
        return message

    def run_training(self,
                     n_steps,
                     batch_size,
                     model_save_path,
                     seed=0):
        
        key = random.PRNGKey(seed)
        data_generator = self.dataset.infinite_data_generator(key, batch_size)

        # TODO support pulling params from elsewhere
        # since we initialised with the same seed, params are the same
        train_state = self.trainer.initial_train_state(key)
        worker_state = self.fl_algo.init_worker_state(train_state)
        
        for i in range(n_steps):
            # Local training
            batch = next(data_generator)
            train_state, loss = self.trainer.update(worker_state.train_state, batch)
            worker_state = WorkerState(train_state=train_state, extras=worker_state.extras)

            # everything below should be run not on the GPU?
            
            if i % 10 == 0:
                test_data_generator = self.dataset.data_generator(key, batch_size)
                metric = self.trainer.eval(worker_state.train_state, test_data_generator)
                self.logger.info(f"Worker {self.worker_id}, Step {i}, Loss {loss:.2f}, Metric {metric:.5f}")

            if self.comms_policy.should_send(i):
                self.logger.info(f"Worker {self.worker_id}, Step {i}, Sending to peers")
                var = self.fl_algo.send_worker_var(worker_state)
                self.broadcast(var)

            if self.comms_policy.should_receive(i,
                    {'n_available_messages': self.comms_manager.n_available_messages}):
                
                messages = self.comms_manager.get_available_messages()
                self.logger.info(f"Worker {self.worker_id}, Step {i}, Receiving {len(messages)} messages")
                worker_state = self.fl_algo.update_worker_state(worker_state, messages)

class FederatedServer(BaseWorker):
    """
    This class handles the server in a centralized setting.
    """
    def __init__(self, worker_id, comms_config, dataset_config, model_config, trainer_config, algo_config):
        super().__init__(worker_id, comms_config)
        self.comms_policy = build_comms_policy(algo_config.server_comms_policy, comms_config)

        self.test_dataset = build_dataset(dataset_config, train=False)

        # TODO could remove this? since we dont need to train on the server
        self.trainer = build_trainer(self.test_dataset, model_config, trainer_config)

        self.fl_algo = build_algo(algo_config.algorithm)

    def run_training(self,
                     n_steps,
                     model_save_path,
                     batch_size,
                     seed=0):
        
        key = random.PRNGKey(seed)

        # TODO support pulling params from elsewhere
        # since we initialised with the same seed, params are the same
        train_state = self.trainer.initial_train_state(key)
        server_state = self.fl_algo.init_server_state(train_state)
        
        counter = 0
        while counter < n_steps:
            # self.logger.info(f"Server {self.worker_id}, Step {counter}, Messages {self.comms_manager.n_available_messages}")
            if self.comms_policy.should_receive(counter, 
                    {'n_available_messages': self.comms_manager.n_available_messages}):
                # receive and update
                messages = self.comms_manager.get_available_messages()
                self.logger.info(f"Server {self.worker_id}, Step {counter}, Received {len(messages)} messages")
                server_state = self.fl_algo.update_server_state(server_state, messages)
                # send back to workers
                var = self.fl_algo.send_server_var(server_state)
                self.broadcast(var)


                # # eval
                # data_generator = self.test_dataset.all_silo_data_generator(key, batch_size)
                # # NOTE if the eval takes too long then we get blocked here
                # metric = self.trainer.eval(server_state.train_state, data_generator)
                # self.logger.info(f"Server, Step {counter}, Metric {metric:.5f}")

                counter += 1

class FederatedEvaluator(BaseWorker):
    """
    This class handles evaluation of the server params.
    Pulls from the server and evaluates on the test dataset.
    """
    def __init__(self, worker_id, comms_config, dataset_config, model_config, trainer_config, algo_config):
        super().__init__(worker_id, comms_config)
        self.comms_policy = build_comms_policy(algo_config.worker_comms_policy, comms_config)
        self.test_dataset = build_dataset(dataset_config, train=False)

        # TODO could remove this? since we dont need to train on the server
        self.trainer = build_trainer(self.test_dataset, model_config, trainer_config)

    def run_training(self,
                     n_steps,
                     model_save_path,
                     batch_size,
                     seed=0):
        
        key = random.PRNGKey(seed)

        # TODO support pulling params from elsewhere
        # since we initialised with the same seed, params are the same
        train_state = self.trainer.initial_train_state(key)
        
        counter = 0
        while True:
            # doesnt need a comms policy since we are just pulling from the server whenever available
            if self.comms_policy.should_receive(counter,
                    {'n_available_messages': self.comms_manager.n_available_messages}):
                
                messages = self.comms_manager.get_available_messages()
                self.logger.info(f"Worker {self.worker_id}, Step {counter}, Receiving {len(messages)} messages")


                train_state = TrainState(params=messages[0].var, opt_state=None)

                # eval
                data_generator = self.test_dataset.all_silo_data_generator(key, batch_size)
                metric = self.trainer.eval(train_state, data_generator)
                self.logger.info(f"Server, Step {counter}, Metric {metric:.5f}")
                counter += 1