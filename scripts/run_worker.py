import hydra
from omegaconf import DictConfig, OmegaConf
from deai.worker import FederatedServer, FederatedWorker, FederatedEvaluator
from functools import partial
from deai.comms import parse_ips, parse_ports

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)

    # ips and ports may be comma-separated strings or lists
    cfg.comms.peer_ports = parse_ports(cfg.comms.peer_ports)
    cfg.comms.peer_ips = parse_ips(cfg.comms.peer_ips)

    if cfg.worker.type == "FederatedServer":
        workertype = FederatedServer
    elif cfg.worker.type == "FederatedEvaluator":
        workertype = FederatedEvaluator
    elif cfg.worker.type == "FederatedWorker":
        workertype = partial(FederatedWorker, dp_config=cfg.dp)
    else:
        raise ValueError(f"Unknown worker type: {cfg.worker.type}")

    worker = workertype(
            worker_id=cfg.worker.worker_id,
            comms_config=cfg.comms,
            dataset_config=cfg.data,
            model_config=cfg.model,
            trainer_config=cfg.trainer,
            algo_config=cfg.algorithm,
    )

    worker.run_training(
        n_steps=cfg.trainer.n_steps,
        batch_size=cfg.trainer.batch_size,
        model_save_path=cfg.trainer.model_save_path,
    )

if __name__ == "__main__":
    my_app()