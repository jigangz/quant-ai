#!/bin/bash
# ===================================
# LocalStack Initialization Script
# ===================================
# Creates SQS queues and SNS topics for local development.
# This runs automatically when LocalStack starts (mounted to init/ready.d/).

set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-us-east-1}"
ENDPOINT="http://localhost:4566"

echo "=== Initializing LocalStack resources ==="

# ===================================
# SQS Queues
# ===================================
echo "Creating SQS queues..."

# Training queue (main job queue)
awslocal sqs create-queue \
    --queue-name quant-ai-training \
    --attributes '{"VisibilityTimeout":"300","MessageRetentionPeriod":"86400"}' \
    --region "$REGION"

# Dead-letter queue for failed training jobs
awslocal sqs create-queue \
    --queue-name quant-ai-training-dlq \
    --attributes '{"MessageRetentionPeriod":"1209600"}' \
    --region "$REGION"

echo "SQS queues created."

# ===================================
# SNS Topics
# ===================================
echo "Creating SNS topics..."

# Alerts topic (price breakout, technical indicators, sentiment)
awslocal sns create-topic \
    --name quant-ai-alerts \
    --region "$REGION"

# Training lifecycle events
awslocal sns create-topic \
    --name quant-ai-training \
    --region "$REGION"

echo "SNS topics created."

# ===================================
# Verify
# ===================================
echo ""
echo "=== LocalStack resources ==="
echo "SQS Queues:"
awslocal sqs list-queues --region "$REGION"
echo ""
echo "SNS Topics:"
awslocal sns list-topics --region "$REGION"
echo ""
echo "=== LocalStack initialization complete ==="
