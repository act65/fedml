#!/bin/bash
python scripts/run_worker.py \
    worker.worker_id=$WORKER_ID \
    worker.type=$WORKER_TYPE \
    comms.ip=$PRIVATE_IP \
    comms.port=$PORT \
    comms.peer_ips="[$PEER_IPS]" \
    comms.peer_ports="[$PEER_PORTS]" \
    data.silo_id=$SILO_ID \
    data=$DATA \
    model=$MODEL \
    trainer.n_steps=$N_STEPS
