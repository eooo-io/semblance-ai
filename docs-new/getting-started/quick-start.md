# Quick Start Guide

## Prerequisites

Before you begin, ensure you have:

- Python 3.8 or higher
- Docker and Docker Compose
- Git
- Node.js 16 or higher (for web interface)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/eooo-io/semblance-ai.git
cd semblance-ai
```

2. Set up the environment:
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. Configure your environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

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

- Read the [Installation Guide](./installation.md) for detailed setup instructions
- Learn about [Configuration](./configuration.md) options
- Explore the [Core Architecture](../core/architecture.md)
- Check out our [Data Curation](../curation/) guide 