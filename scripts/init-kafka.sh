#!/usr/bin/env bash
# ===================================
# Initialize Kafka Topics
# ===================================
# Run after Kafka broker is ready (used by docker-compose).
# Usage: ./scripts/init-kafka.sh [BOOTSTRAP_SERVERS]

set -euo pipefail

BOOTSTRAP="${1:-localhost:9092}"
KAFKA_BIN="${KAFKA_BIN:-/opt/bitnami/kafka/bin}"

echo "Waiting for Kafka at ${BOOTSTRAP}..."
for i in $(seq 1 30); do
    if "${KAFKA_BIN}/kafka-topics.sh" --bootstrap-server "${BOOTSTRAP}" --list >/dev/null 2>&1; then
        echo "Kafka is ready."
        break
    fi
    echo "  attempt ${i}/30..."
    sleep 2
done

TOPICS=(
    "market.prices"
    "news.raw"
    "news.scored"
    "signals.generated"
)

for topic in "${TOPICS[@]}"; do
    echo "Creating topic: ${topic}"
    "${KAFKA_BIN}/kafka-topics.sh" \
        --bootstrap-server "${BOOTSTRAP}" \
        --create \
        --if-not-exists \
        --topic "${topic}" \
        --partitions 3 \
        --replication-factor 1
done

echo "All Kafka topics created."
"${KAFKA_BIN}/kafka-topics.sh" --bootstrap-server "${BOOTSTRAP}" --list
