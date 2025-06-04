# API Reference

## Overview

The Semblance AI API is organized around REST. Our API accepts JSON-encoded request bodies, returns JSON-encoded responses, and uses standard HTTP response codes, authentication, and verbs.

## Base URLs

- **Production**: `https://api.semblance.ai/v1`
- **Staging**: `https://api.staging.semblance.ai/v1`
- **Development**: `http://localhost:8000/v1`

## Authentication

```bash
curl -X GET "https://api.semblance.ai/v1/models" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

All API requests require authentication using Bearer tokens. Include your API key in the Authorization header:

## Core API

### Models

#### List Models

```http
GET /models
```

Lists available models.

**Response**
```json
{
  "models": [
    {
      "id": "semblance-1b",
      "name": "Semblance 1B",
      "version": "1.0.0",
      "status": "ready"
    }
  ]
}
```

#### Get Model Details

```http
GET /models/{model_id}
```

Get details about a specific model.

### Training

#### Create Training Job

```http
POST /training/jobs
```

Create a new training job.

**Request Body**
```json
{
  "model_id": "semblance-1b",
  "dataset_id": "personal-dataset-001",
  "config": {
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5
  }
}
```

#### List Training Jobs

```http
GET /training/jobs
```

List all training jobs.

### Inference

#### Generate Response

```http
POST /generate
```

Generate a response from the model.

**Request Body**
```json
{
  "model": "semblance-1b",
  "prompt": "What are your thoughts on...",
  "max_tokens": 100,
  "temperature": 0.7
}
```

## RAG API

### Documents

#### Index Document

```http
POST /rag/documents
```

Index a new document.

**Request Body**
```json
{
  "content": "Document content...",
  "metadata": {
    "title": "Example Document",
    "author": "John Doe",
    "date": "2024-03-21"
  }
}
```

#### Search Documents

```http
GET /rag/documents/search
```

Search indexed documents.

**Query Parameters**
- `q`: Search query
- `limit`: Maximum results (default: 10)
- `offset`: Result offset (default: 0)

### Queries

#### Process Query

```http
POST /rag/query
```

Process a RAG query.

**Request Body**
```json
{
  "query": "What is...",
  "top_k": 5,
  "options": {
    "rerank": true,
    "filter": {
      "date_range": {
        "start": "2024-01-01",
        "end": "2024-12-31"
      }
    }
  }
}
```

## Curation API

### Datasets

#### Create Dataset

```http
POST /curation/datasets
```

Create a new dataset.

**Request Body**
```json
{
  "name": "Personal Knowledge Base",
  "description": "My curated knowledge",
  "schema": {
    "fields": [
      {
        "name": "text",
        "type": "string",
        "required": true
      }
    ]
  }
}
```

#### Add Data

```http
POST /curation/datasets/{dataset_id}/items
```

Add items to a dataset.

### Annotations

#### Create Annotation Task

```http
POST /curation/tasks
```

Create a new annotation task.

**Request Body**
```json
{
  "dataset_id": "dataset-001",
  "type": "text_classification",
  "config": {
    "labels": ["positive", "negative", "neutral"]
  }
}
```

## Error Handling

### HTTP Status Codes

- `200`: Success
- `201`: Created
- `400`: Bad Request
- `401`: Unauthorized
- `403`: Forbidden
- `404`: Not Found
- `429`: Too Many Requests
- `500`: Internal Server Error

### Error Response Format

```json
{
  "error": {
    "code": "invalid_request",
    "message": "Invalid request parameters",
    "details": {
      "field": "temperature",
      "issue": "Must be between 0 and 1"
    }
  }
}
```

## Rate Limits

- Free tier: 60 requests per minute
- Pro tier: 1000 requests per minute
- Enterprise: Custom limits

Rate limit headers are included in all responses:
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1616179200
```

## SDKs & Libraries

Official SDKs are available for:

- Python: `pip install semblance-ai`
- JavaScript: `npm install @semblance/sdk`
- Go: `go get github.com/eooo-io/semblance-go`

Example Python usage:
```python
from semblance import Client

client = Client("YOUR_API_KEY")

response = client.generate(
    model="semblance-1b",
    prompt="Tell me about...",
    max_tokens=100
)
```

## Webhooks

Configure webhooks to receive real-time updates:

```http
POST /webhooks
```

**Request Body**
```json
{
  "url": "https://your-server.com/webhook",
  "events": ["training.completed", "inference.failed"],
  "secret": "your-signing-secret"
}
```

## Best Practices

1. **Error Handling**
   - Always check HTTP status codes
   - Implement exponential backoff for retries
   - Handle rate limits gracefully

2. **Performance**
   - Use connection pooling
   - Implement caching where appropriate
   - Batch requests when possible

3. **Security**
   - Store API keys securely
   - Use HTTPS for all requests
   - Validate webhook signatures

## Support

- [API Status](https://status.semblance.ai)
- [Developer Discord](https://discord.gg/semblance-ai)
- [GitHub Issues](https://github.com/eooo-io/semblance-ai/issues) 