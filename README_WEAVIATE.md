# AI Agent ETL Pipeline with Weaviate Integration

This extension of the AI Agent ETL Pipeline adds Weaviate vector database integration for efficient semantic search across both text and image embeddings.

## Features

- Store text and image embeddings in Weaviate vector database
- Support for both local Weaviate instance and Weaviate Cloud Service
- Semantic search across text content and images
- Docker configuration for easy deployment

## Prerequisites

- Docker and Docker Compose for running the local setup
- A Weaviate Cloud account (optional, for cloud deployment)
- API keys for your preferred embedding provider (Cohere or OpenAI)

## Getting Started

### Option 1: Using a Local Weaviate Instance (for Development/Testing)

1. Start the ETL pipeline with a local Weaviate instance:

```bash
docker-compose -f docker-compose-weaviate.yml up -d
```

This will:
- Start a local Weaviate instance
- Run the ETL processor to extract and process data
- Import the processed data into Weaviate

2. Run a search query:

```bash
docker exec -it ai_agent_etl_pipeline-etl-importer-1 python query_with_weaviate.py "your search query"
```

### Option 2: Using Weaviate Cloud Service (for Production)

1. Create a `.env` file with your credentials:

```
WEAVIATE_URL=https://your-cluster-url.weaviate.cloud
WEAVIATE_API_KEY=your-weaviate-api-key
COHERE_API_KEY=your-cohere-api-key  # Or OPENAI_API_KEY
```

2. Update the environment section in the `docker-compose-weaviate.yml` file:

```yaml
environment:
  - PYTHONUNBUFFERED=1
  - WEAVIATE_URL=${WEAVIATE_URL}
  - WEAVIATE_API_KEY=${WEAVIATE_API_KEY}
  - COHERE_API_KEY=${COHERE_API_KEY}
```

3. Start the ETL pipeline:

```bash
docker-compose -f docker-compose-weaviate.yml up -d etl-processor etl-importer
```

4. Run a search query:

```bash
docker exec -it ai_agent_etl_pipeline-etl-importer-1 python query_with_weaviate.py "your search query"
```

## Running Directly (Without Docker)

You can also run the scripts directly without Docker:

1. Install the required dependencies:

```bash
pip install -r requirements_weaviate.txt
```

2. Set environment variables:

```bash
export WEAVIATE_URL=https://your-cluster-url.weaviate.cloud
export WEAVIATE_API_KEY=your-weaviate-api-key
export COHERE_API_KEY=your-cohere-api-key
```

3. Process the data:

```bash
python bulk_process_files_with_phoenix.py
```

4. Import the processed data into Weaviate:

```bash
python weaviate_client.py
```

5. Run a search query:

```bash
python query_with_weaviate.py "your search query"
```

## Configuration Options

### Environment Variables

- `WEAVIATE_URL`: URL of the Weaviate instance or cluster
- `WEAVIATE_API_KEY`: API key for Weaviate Cloud Service
- `COHERE_API_KEY`: API key for Cohere embeddings
- `OPENAI_API_KEY`: API key for OpenAI embeddings (alternative to Cohere)

### Command Line Arguments

The `query_with_weaviate.py` script supports several command-line arguments:

```
python query_with_weaviate.py "your search query" --top-k 10 --cluster-url https://your-cluster-url.weaviate.cloud --api-key your-api-key --embedding-api-key your-embedding-api-key
```

- `query`: The search query (required)
- `--top-k`: Number of top results to return (default: 5)
- `--cluster-url`: URL of Weaviate Cloud cluster
- `--api-key`: API key for Weaviate Cloud
- `--embedding-api-key`: API key for embedding provider (Cohere or OpenAI)

## Development and Customization

### Embedding Models

By default, this implementation uses mock embeddings for demonstration purposes. In a production environment, you would want to replace the `get_mock_embedding` function with a real embedding model.

You can modify the `weaviate_client.py` file to use a different embedding model or provider:

```python
# Example using OpenAI embeddings
import openai

def get_real_embedding(text: str) -> List[float]:
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']
```

### Collection Schema

The schema for the collections is defined in the `_create_schema` method of the `WeaviateClient` class. You can modify this to add additional properties or collections as needed.

## Troubleshooting

- **Connection Issues**: Ensure your Weaviate cluster URL and API key are correct
- **Missing Embeddings**: Check that your embedding API key is set correctly
- **Empty Results**: Make sure your ETL pipeline has processed the data before searching