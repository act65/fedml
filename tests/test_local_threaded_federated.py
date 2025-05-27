import hydra
from omegaconf import DictConfig, OmegaConf

import threading
from deai.worker import FederatedServer, FederatedWorker, FederatedEvaluator
from deai.network import construct_centralized
from deai.logging import FileLogger, error_handling_decorator

import logging
logging.getLogger('werkzeug').setLevel(logging.WARNING)

from functools import partial

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def my_app(cfg : DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))

    ips = ["localhost", "localhost", "localhost", "localhost"]
    ports = [5000, 5001, 5002, 5003]

    network = construct_centralized(ips, ports)
    
    workers = []
    counter = 0

    for i in range(len(ips)):
        comms_config = DictConfig({
                'ip': cfg.comms.ip,
                'port': network.get_node_port(i),
                'peer_ips': network.get_peer_ips(i),
                'peer_ports': network.get_peer_ports(i)
            })
        
        if network.get_node_type(i) == 'FederatedServer':
            workertype = FederatedServer
        elif network.get_node_type(i) == 'FederatedEvaluator':
            workertype = FederatedEvaluator
        elif network.get_node_type(i) == 'FederatedWorker':
            workertype = partial(FederatedWorker, dp_config=cfg.dp)
        else:
            raise ValueError(f"Unknown worker type: {network.get_node_type(i)}")
        

        cfg.data.silo_id = counter if network.get_node_type(i) == 'FederatedWorker' else 2
        counter += 1 if network.get_node_type(i) == 'FederatedWorker' else 0

        workers.append(workertype(
            worker_id=i,
            comms_config=comms_config,
            dataset_config=cfg.data,
            model_config=cfg.model,
            trainer_config=cfg.trainer,
            algo_config=cfg.algorithm,
        ))

    # Start training on all workers
    for w in workers:
        threading.Thread(target=error_handling_decorator(w.run_training, FileLogger(f"logs/err-worker-{w.worker_id}")),
            kwargs={
                'n_steps': cfg.trainer.n_steps,
                'batch_size': cfg.trainer.batch_size,
                'model_save_path': '/tmp/test',
            }
        ).start()

if __name__ == "__main__":
    my_app()