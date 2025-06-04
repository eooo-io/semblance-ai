# Deployment Guide

This guide covers deployment options and best practices for the Semblance AI system.

## Prerequisites

Before deploying, ensure you have:
1. Completed the [configuration setup](/getting-started/configuration)
2. Reviewed the [security guidelines](/development/security)
3. Set up [monitoring](/development/monitoring)

## Deployment Options

### Local Deployment
For development and testing purposes.

### Cloud Deployment
For production environments.

### Edge Deployment
For specialized use cases requiring local inference.

## Scaling
See the [scaling guide](/development/scaling) for detailed information about scaling your deployment.

## Security
Review the [security documentation](/development/security) for deployment security best practices.

## Monitoring
Set up proper monitoring using our [monitoring guide](/development/monitoring).

## Overview

This guide covers deploying Semblance AI in various environments, from development to production. We support multiple deployment options to suit different needs and scales.

## Deployment Options

### Local Development
- Docker Compose setup
- Direct installation
- Development tools

### Cloud Deployment
- Kubernetes
- Cloud-managed services
- Hybrid setups

### Edge Deployment
- Single machine
- GPU acceleration
- Resource optimization

## Prerequisites

### System Requirements

#### Minimum (Development)
- 16GB RAM
- 4 CPU cores
- 100GB storage
- NVIDIA GPU (optional)

#### Recommended (Production)
- 32GB+ RAM
- 8+ CPU cores
- 500GB+ SSD storage
- NVIDIA GPU with 8GB+ VRAM
- Ubuntu 20.04+ or similar

### Software Requirements
- Docker Engine 24.0.0+
- Docker Compose v2.20.0+
- Kubernetes 1.24+ (for cloud deployment)
- NVIDIA drivers (for GPU support)
- Python 3.9+

## Local Deployment

### Using Docker Compose

1. Clone the repository:
```bash
git clone https://github.com/eooo-io/semblance-ai.git
cd semblance-ai
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

3. Start services:
```bash
# Development mode
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Production mode
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Direct Installation

1. Set up Python environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

2. Install dependencies:
```bash
# Ubuntu/Debian
apt-get update && apt-get install -y \
    postgresql-client \
    redis-tools \
    nvidia-container-toolkit

# macOS
brew install postgresql redis
```

3. Start services:
```bash
# Core services
./scripts/start-services.sh

# Web interface
cd webapp && npm install && npm run dev
```

## Cloud Deployment

### Kubernetes Deployment

1. Configure Kubernetes:
```bash
# Create namespace
kubectl create namespace semblance

# Apply configurations
kubectl apply -f k8s/
```

2. Set up monitoring:
```bash
# Install monitoring stack
helm install monitoring prometheus-community/kube-prometheus-stack

# Configure Grafana
kubectl apply -f k8s/monitoring/
```

3. Deploy services:
```bash
# Deploy core services
kubectl apply -f k8s/core/

# Deploy RAG system
kubectl apply -f k8s/rag/

# Deploy curation system
kubectl apply -f k8s/curation/
```

### Cloud Provider Setup

#### AWS

```yaml
# EKS cluster configuration
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: semblance-cluster
  region: us-west-2

nodeGroups:
  - name: cpu-workers
    instanceType: c5.2xlarge
    desiredCapacity: 3
    
  - name: gpu-workers
    instanceType: g4dn.xlarge
    desiredCapacity: 2
    labels:
      nvidia.com/gpu: "true"
```

#### Google Cloud

```bash
# Create GKE cluster
gcloud container clusters create semblance-cluster \
    --machine-type n1-standard-4 \
    --num-nodes 3 \
    --zone us-central1-a \
    --cluster-version latest
```

#### Azure

```bash
# Create AKS cluster
az aks create \
    --resource-group semblance-group \
    --name semblance-cluster \
    --node-count 3 \
    --enable-addons monitoring \
    --generate-ssh-keys
```

## Edge Deployment

### Single Machine Setup

1. Install requirements:
```bash
# Install NVIDIA drivers and CUDA
sudo apt-get update
sudo apt-get install -y nvidia-driver-535 nvidia-cuda-toolkit
```

2. Configure services:
```bash
# Set up systemd services
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable semblance-core.service
```

3. Start services:
```bash
sudo systemctl start semblance-core.service
sudo systemctl start semblance-rag.service
sudo systemctl start semblance-curation.service
```

### Resource Optimization

```yaml
# Resource limits configuration
resources:
  core:
    cpu: 2
    memory: 8Gi
    gpu: 1
  rag:
    cpu: 1
    memory: 4Gi
  curation:
    cpu: 1
    memory: 4Gi
```

## Monitoring & Maintenance

### Health Checks

```bash
# Check service status
curl http://localhost:8000/health

# Monitor resources
docker stats

# Check logs
kubectl logs -f deployment/semblance-core
```

### Backup & Recovery

1. Database backup:
```bash
# Backup PostgreSQL
pg_dump -U postgres semblance > backup.sql

# Backup vector store
weaviate-backup create --output backup/
```

2. Model backup:
```bash
# Export models
mlflow models export-all

# Backup training data
dvc push
```

## Security Considerations

### Network Security

```nginx
# Nginx configuration
server {
    listen 443 ssl;
    server_name api.semblance.ai;

    ssl_certificate /etc/letsencrypt/live/api.semblance.ai/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.semblance.ai/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Access Control

```yaml
# RBAC configuration
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: semblance-role
rules:
  - apiGroups: [""]
    resources: ["pods", "services"]
    verbs: ["get", "list", "watch"]
```

## Troubleshooting

### Common Issues

1. GPU not detected:
```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker GPU support
docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
```

2. Service connectivity:
```bash
# Test database connection
psql -h localhost -U postgres -d semblance

# Test Redis connection
redis-cli ping
```

### Debug Tools

```bash
# Enable debug logging
export DEBUG=1

# Collect diagnostics
./scripts/collect-diagnostics.sh

# Monitor metrics
curl http://localhost:9090/metrics
```

## Next Steps

- [Configuration Guide](../getting-started/configuration.md)
- [Monitoring Guide](./monitoring.md)
- [Scaling Guide](./scaling.md)
- [Security Guide](./security.md) 