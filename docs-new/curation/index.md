# Data Curation System

The Data Curation System provides tools and interfaces for collecting, annotating, and managing training data.

## Services

The system consists of several services that work together:

- **Web Interface**: Available at `http://localhost:3000` in development
- **Label Studio**: Available at `http://localhost:8080` in development
- **Jupyter Lab**: Available at `http://localhost:8888` in development
- **MinIO**: Available at `http://localhost:9001` in development

## Getting Started

1. Follow the [installation guide](/getting-started/installation) to set up the system
2. Configure your environment using the [configuration guide](/getting-started/configuration)
3. Learn about [deployment options](/curation/deployment)

## Development

For development setup and contribution guidelines, see:

- [Local Development Setup](/development/local-setup)
- [Contributing Guidelines](/development/contributing)
- [API Reference](/curation/api-reference)

## Architecture

The curation system uses a microservices architecture with the following components:

- Web UI (React + TypeScript)
- Label Studio for annotation
- MinIO for object storage
- PostgreSQL for metadata
- Redis for caching

For detailed architecture information, see the [Architecture Guide](/curation/architecture).

## Security

For security best practices and configuration, see the [Security Guide](/curation/security). 