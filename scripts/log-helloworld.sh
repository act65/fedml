#!/bin/sh
set -e

LOG_FILE="/var/log/helloworld.log"
TIMESTAMP=$(date +'%Y-%m-%d %H:%M:%S')

# Try to get AWS metadata (with timeout for local environments)
get_aws_metadata() {
    curl --max-time 2 -s "http://169.254.169.254/latest/meta-data/$1" 2>/dev/null || echo "non-aws"
}

INSTANCE_ID=$(get_aws_metadata "instance-id")
AWS_REGION=$(get_aws_metadata "placement/region")

echo "${TIMESTAMP} - HelloWorld - Instance: ${INSTANCE_ID} - Region: ${AWS_REGION}" >> ${LOG_FILE}

# Keep container running
tail -f /dev/null
