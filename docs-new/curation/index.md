# Data Curation

Semblance Curation is a comprehensive platform designed for building, maintaining, and curating machine learning datasets. It provides an end-to-end solution for data collection, annotation, preprocessing, and quality control, with built-in support for multi-modal data types.

## Tech Stack

### Core Components
- **Label Studio** - Advanced data annotation
- **Argilla** - Data quality management
- **Weaviate** - Vector search and storage
- **Ollama** - Local model inference

### Data Storage
- **PostgreSQL** - Structured data storage
- **Redis** - High-performance caching
- **Elasticsearch** - Text search and analytics
- **MinIO** - Object storage system

### Development & ML Tools
- **Jupyter** - Interactive development
- **MLflow** - Experiment tracking
- **Ray** - Distributed computing

### Monitoring & Observability
- **Prometheus** - Metrics collection
- **Grafana** - Visualization dashboard
- **Loki** - Log aggregation
- **Jaeger** - Distributed tracing

## System Requirements

### Minimum Requirements
- 32GB RAM
- 8+ CPU cores
- NVIDIA GPU with 8GB+ VRAM (recommended)
- 500GB+ SSD storage
- Ubuntu 20.04+ or similar Linux distribution

## Key Features

### Multi-modal Data Handling
- Process text, voice, and video data
- Scalable storage solutions
- Efficient annotation workflows
- Format conversion utilities

### Machine Learning Operations
- Local LLM inference with Ollama
- GPU-accelerated processing
- Vector-based similarity search
- Comprehensive annotation tools

### Advanced Data Management
- Powerful text search
- Structured data storage
- High-performance caching
- Data versioning and lineage
- Automated quality checks
- Real-time monitoring

### Development Environment
- Interactive Jupyter notebooks
- Data science toolkit
- Containerized architecture
- Full GPU support
- Distributed training
- Experiment tracking

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/eooo-io/semblance-curation.git
cd semblance-curation
```

2. Configure environment:
```bash
cp env-example .env
# Edit .env with your settings
```

3. Start services:
```bash
# For production
docker compose up -d

# For development
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

4. Access interfaces:
- Label Studio: http://localhost:8080
- Jupyter Lab: http://localhost:8888
- MinIO Console: http://localhost:9001
- Grafana: http://localhost:3000

## Optional Components

### MLflow Integration
```yaml
# Add to docker-compose.yml
mlflow:
  image: ghcr.io/mlflow/mlflow:latest
  ports:
    - "5000:5000"
  environment:
    - MLFLOW_TRACKING_URI=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
  depends_on:
    - postgres
```

### Ray Cluster
```yaml
# Add to docker-compose.yml
ray-head:
  image: rayproject/ray:latest
  ports:
    - "8265:8265"  # Dashboard
    - "10001:10001"  # Client server
  command: ray start --head --dashboard-host=0.0.0.0
```

For detailed setup instructions and advanced configuration options, see:
- [Installation Guide](./installation.md)
- [Configuration Guide](./configuration.md)
- [Deployment Guide](./deployment.md) 