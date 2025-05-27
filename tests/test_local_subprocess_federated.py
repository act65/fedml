from subprocess import Popen, PIPE, STDOUT
from deai.network import construct_centralized

def main():
    ips = ["localhost", "localhost", "localhost", "localhost"]
    ports = [5000, 5001, 5002, 5003]

    network = construct_centralized(ips, ports)
    
    processes = []
    counter = 0
    for i in range(len(ips)):
        WORKER_INDEX = i
        PORT = network.get_node_port(i)
        PRIVATE_IP = network.get_node_ip(i)
        PEER_IPS = network.get_peer_ips(i)
        PEER_PORTS = network.get_peer_ports(i)
        WORKER_TYPE = network.get_node_type(i)
        
        cmd = ["python", "scripts/run_worker.py",
            f"worker.worker_id={WORKER_INDEX}",
            f"worker.type={WORKER_TYPE}",
            f"comms.ip={PRIVATE_IP}",
            f"comms.port={PORT}",
            f"comms.peer_ips=[{','.join([str(ip) for ip in PEER_IPS])}]",
            f"comms.peer_ports=[{','.join([str(port) for port in PEER_PORTS])}]",
            f"data.silo_id={counter if WORKER_TYPE == 'FederatedWorker' else None}",
            "data=gfemnist",
            "model=cnn-small",
            # f"data=fesimple",
            # f"model=mlp-med",
            f"trainer.n_steps=200"
            ]

        counter += 1 if WORKER_TYPE == 'FederatedWorker' else 0

        print(' '.join(cmd))
        processes.append(Popen(cmd, stdout=PIPE, stdin=PIPE, stderr=PIPE, text=True))

    for p in processes:
        stdout_output, stderr_output = p.communicate()
        print("stdout:", stdout_output)
        print("stderr:", stderr_output)


if __name__ == "__main__":
    main()