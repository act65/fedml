#!/bin/bash
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URL="$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URL

docker build -t my-hello-world .
docker tag my-hello-world:latest $ECR_URL/my-hello-world:latest
docker push $ECR_URL/my-hello-world:latest