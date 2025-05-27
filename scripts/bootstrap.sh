#!/bin/bash

# Update package lists and install Docker
apt update
apt install -y docker.io

# Pull the Docker image
docker pull your-registry/decentralized-worker:latest

# Get the instance ID
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)

# Get the private IP address of the instance
PRIVATE_IP=$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4)

# Extract the worker index from the instance name
WORKER_INDEX=$(echo $(aws ec2 describe-instances --instance-ids $INSTANCE_ID --query 'Reservations[*].Instances[*].Tags[?Key==`Name`].Value' --output text) | grep -oE '[0-9]+')

# Discover peer IPs using tags
PEER_IPS=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=decentralized-worker*" \ # Filter by tag
    --query 'Reservations[*].Instances[*].PrivateIpAddress' \ # Get private IPs
    --output text | tr '\n' ',' | sed 's/,$//') # Format as comma-separated string

# Run the Docker container
docker run -d \
    -p 5000:5000 \
    -e WORKER_INDEX=$WORKER_INDEX \
    -e PRIVATE_IP=$(curl -s http://169.254.169.254/latest/meta-data/local-ipv4) \
    -e PEER_IPS="$PEER_IPS" \ # Pass peer IPs
    --name decentralized-worker decentralized-worker:latest decentralized-worker:latest

# You might need to configure networking between containers/hosts depending on your setup.