# Environment Setup Guide

This document explains how to set up the Quant-AI development environment.

## Quick Start (Docker)

The fastest way to get started:

```bash
# 1. Clone the repo
git clone https://github.com/jigangz/quant-ai.git
cd quant-ai

# 2. Copy environment file
cp .env.example .env

# 3. Start services
docker-compose up

# 4. Verify
curl http://localhost:8000/health
```

## Local Development (without Docker)

### Prerequisites

- Python 3.10+
- PostgreSQL 15+ (optional, can use SQLite)

### Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment file
cp .env.example .env

# 4. Edit .env for your setup
# For quick testing, use SQLite:
# DATABASE_URL=sqlite:///./quant.db

# 5. Run the server
uvicorn app.main:app --reload

# 6. Verify
curl http://localhost:8000/health
```

## Environment Variables

### Required

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | `sqlite:///:memory:` |

### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `ENV` | Environment (dev/prod/test) | `dev` |
| `DEBUG` | Enable debug mode | `false` |
| `PROVIDERS_ENABLED` | Enabled data providers | `market` |
| `DEFAULT_FEATURE_GROUPS` | Default feature groups | `ta_basic,volatility` |
| `DEFAULT_MODEL_TYPE` | Default model type | `logistic` |
| `STORAGE_BACKEND` | Artifact storage (local/supabase/s3) | `local` |

### Supabase (Production)

For production model registry:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-key  # For admin operations
```

### S3 Storage (Production)

For storing model artifacts in S3:

```env
STORAGE_BACKEND=s3
S3_BUCKET=your-bucket
S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
```

## GPU Setup (Optional)

For GPU-accelerated training (XGBoost, LightGBM, future deep learning):

### Ubuntu + NVIDIA

```bash
# 1. Install NVIDIA driver
sudo apt-get update
sudo apt-get install -y nvidia-driver-535

# 2. Install CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-2

# 3. Verify
nvidia-smi
```

### Docker with GPU

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Run with GPU
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

## Verify Setup

After setup, run:

```bash
# Check health endpoint
curl http://localhost:8000/health

# Expected response:
{
  "status": "ok",
  "settings": {
    "env": "dev",
    "debug": true,
    "providers_enabled": ["market"],
    "default_feature_groups": ["ta_basic", "volatility"],
    "default_model_type": "logistic",
    ...
  }
}
```

## Troubleshooting

### Database Connection Error

```
sqlalchemy.exc.OperationalError: could not connect to server
```

**Solution:** Ensure PostgreSQL is running, or use SQLite for local testing:
```env
DATABASE_URL=sqlite:///./quant.db
```

### Port Already in Use

```
OSError: [Errno 98] Address already in use
```

**Solution:** Change the port:
```bash
uvicorn app.main:app --reload --port 8001
```

### Docker Build Fails

```
ERROR: failed to solve: python:3.11-slim: not found
```

**Solution:** Pull the image manually:
```bash
docker pull python:3.11-slim
```
