# Quick Start Guide

Get started with Semblance AI in minutes.

## Prerequisites

Before you begin, ensure you have:

- Python 3.9+
- Docker and Docker Compose
- Git
- Node.js 16 or higher (for web interface)

## Installation

1. Follow the [installation guide](/getting-started/installation)
2. Configure your environment using the [configuration guide](/getting-started/configuration)

## Basic Usage

### Data Curation

1. Start the data curation service:
```bash
docker-compose -f components/curation/docker-compose.yml up -d
```

2. Access the Label Studio interface at `http://localhost:8080`

### RAG System

1. Start the RAG system:
```bash
docker-compose -f components/rag/docker-compose.yml up -d
```

2. Access the web interface at `http://localhost:3000`

## Next Steps

- [Core Concepts](/core/concepts)
- [Architecture Overview](/core/architecture)
- [Development Guide](/development/contributing) 