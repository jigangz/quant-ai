# Quant AI — Kubernetes Deployment

Local Minikube deployment for the Quant AI distributed stack.

## Prerequisites

- [Minikube](https://minikube.sigs.k8s.io/docs/start/) (>= 1.32)
- [kubectl](https://kubernetes.io/docs/tasks/tools/)
- Docker Desktop (or Docker Engine) — Minikube uses Docker as its driver by default

## One-time setup

```bash
# Start Minikube with 4 CPUs, 6GB RAM (HPA needs headroom)
minikube start --cpus=4 --memory=6g

# Enable metrics-server addon (required for HPA to work)
minikube addons enable metrics-server

# Point local Docker client at Minikube's Docker daemon so
# images built locally are visible inside the cluster
eval $(minikube docker-env)
```

## Build images

```bash
# From repository root
docker build -t quant-ai:latest -f Dockerfile --target production .
docker build -t quant-ai-consumer:latest -f Dockerfile.consumer .
```

## Deploy

```bash
# Copy secret template and fill in values (at minimum DATABASE_URL)
cp k8s/secret.example.yaml k8s/secret.yaml
# Edit k8s/secret.yaml with your values — see secret.example.yaml for keys

# Create Grafana dashboard ConfigMap from JSON file
kubectl create namespace quant-ai --dry-run=client -o yaml | kubectl apply -f -
kubectl -n quant-ai create configmap grafana-dashboards \
    --from-file=observability/dashboards/quant-ai.json \
    --dry-run=client -o yaml | kubectl apply -f -

# Apply all manifests in one shot
kubectl apply -f k8s/
```

## Verify

```bash
# All pods should be Running/Ready
kubectl -n quant-ai get pods
# Expect: api (2), consumer (1), kafka, postgres, redis, prometheus, grafana — all Running

# HPA status
kubectl -n quant-ai get hpa
# Expect: quant-ai-api target current/70%

# Service endpoints (Minikube NodePort)
minikube service -n quant-ai api            # API on :30001
minikube service -n quant-ai consumer       # Consumer on :30002
minikube service -n quant-ai grafana        # Grafana on :30030 (login admin/admin)
minikube service -n quant-ai prometheus     # Prometheus UI on :30090

# Confirm all deployments are available
kubectl -n quant-ai get deployments
kubectl -n quant-ai get statefulsets
```

## Smoke test

```bash
API_URL=$(minikube -n quant-ai service api --url)
CONSUMER_URL=$(minikube -n quant-ai service consumer --url)

# Health checks
curl "${API_URL}/health"
# Expect: {"status":"ok"}

curl "${API_URL}/health/ready"
# Expect: {"status":"ready"}

curl "${CONSUMER_URL}/health"
# Expect: {"status":"ok"}

# Hit predict endpoint 10 times to generate Kafka events
for i in {1..10}; do
    curl -s "${API_URL}/predict?ticker=AAPL&lookback=10" > /dev/null
done

# Confirm consumer aggregated events via stats endpoint
curl "${CONSUMER_URL}/stats/AAPL"
# Expect: {"ticker":"AAPL","count":10,"avg_confidence":...,"bullish_ratio":...}

# Check Prometheus metrics endpoint
curl "${API_URL}/metrics" | grep "quant_ai_predictions"
# Expect: quant_ai_predictions_total{...} count

# Grafana login
# Open http://$(minikube -n quant-ai service grafana --url)
# Login: admin / admin
# Navigate to Dashboards → Quant AI Dashboard
```

## Tear down

```bash
# Delete all resources in the namespace
kubectl delete namespace quant-ai

# Or nuke the entire Minikube cluster
minikube delete
```

## Manifest inventory

| File | Kind | Purpose |
|------|------|---------|
| `namespace.yaml` | Namespace | Isolates all resources in `quant-ai` |
| `configmap.yaml` | ConfigMap | Non-secret env vars (ENV, BROKER_BACKEND, etc.) |
| `secret.example.yaml` | Secret (template) | DATABASE_URL, Supabase keys |
| `deployment-api.yaml` | Deployment | API service (2 replicas, HPA-managed) |
| `service-api.yaml` | Service (NodePort 30001) | Exposes API outside cluster |
| `hpa-api.yaml` | HorizontalPodAutoscaler | Scale api 2–5 at CPU >70% |
| `deployment-consumer.yaml` | Deployment | Events consumer (1 replica) |
| `service-consumer.yaml` | Service (NodePort 30002) | Exposes consumer outside cluster |
| `statefulset-kafka.yaml` | StatefulSet | Kafka broker (KRaft, 1Gi PVC) |
| `service-kafka.yaml` | Service (headless) | Kafka internal DNS |
| `statefulset-postgres.yaml` | StatefulSet | Postgres (1Gi PVC) |
| `service-postgres.yaml` | Service (ClusterIP) | Postgres internal access |
| `deployment-redis.yaml` | Deployment | Redis cache |
| `service-redis.yaml` | Service (ClusterIP) | Redis internal access |
| `configmap-prometheus.yaml` | ConfigMap | Prometheus scrape config |
| `deployment-prometheus.yaml` | Deployment + Service | Prometheus (NodePort 30090) |
| `configmap-grafana.yaml` | ConfigMap | Grafana datasource + dashboard provider |
| `deployment-grafana.yaml` | Deployment + Service | Grafana (NodePort 30030) |

## Scaling to cloud (future)

- Replace `statefulset-postgres.yaml` with the Supabase connection in `secret.yaml` — add `DATABASE_URL` pointing to Supabase pooler
- Replace `statefulset-kafka.yaml` with Confluent Cloud credentials (add SASL env vars: `KAFKA_SASL_USERNAME`, `KAFKA_SASL_PASSWORD`)
- Change `image: quant-ai:latest` to a registry-pushed image (ECR / GCR / DockerHub)
- Swap NodePort services for LoadBalancer + Ingress controller with TLS termination
- Add `PodDisruptionBudget` for api to guarantee rolling deploys don't drop below 1 available pod
- Use `ExternalSecret` operator or Sealed Secrets for production secret management
