import hydra
from omegaconf import DictConfig, OmegaConf

import threading
from deai.worker import DecentralizedWorker

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    all_peer_ports = cfg.worker.peer_ports + [cfg.worker.port]
    
    # all-to-all topology
    all_to_all_peers = {p: [p2 for p2 in all_peer_ports if p2 != p]
                for p in all_peer_ports}
    print(all_to_all_peers)
    
    workers = []
    for i, p in enumerate(all_peer_ports):
        dt = DecentralizedWorker(
            worker_id=i,
            ip=cfg.worker.ip,
            port=p,
            peer_ips=[cfg.worker.ip]*len(all_peer_ports),
            peer_ports=all_to_all_peers[p],
            dataset_config=cfg.data,
            network_config=cfg.network,
            trainer_config=cfg.trainer,
        )
        workers.append(dt)

    print(workers)
    
    # Start training on all workers
    for w in workers:
        threading.Thread(target=w.run_training, 
            kwargs={
                'n_steps': cfg.trainer.n_steps,
                'batch_size': cfg.trainer.batch_size,
                'model_save_path': '/tmp/test',
            }
        ).start()

if __name__ == "__main__":
    my_app()