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

This implementation now supports both OpenAI embeddings and fallback mock embeddings. When an OpenAI API key is provided through the `OPENAI_API_KEY` environment variable, the system will use the OpenAI `text-embedding-3-small` model to generate embeddings. If no API key is available, it automatically falls back to mock embeddings for demonstration purposes.

The implementation is in the `get_embedding` function in `bulk_process_files_with_phoenix.py`:

```python
# Current implementation using OpenAI embeddings with fallback
import openai
import os

def get_embedding(text: str) -> List[float]:
    # Try to use OpenAI's API if API key is available
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        client = openai.OpenAI(api_key=openai_api_key)
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    # Fallback to mock embeddings if no API key
    return mock_embedding(text)
```

### Collection Schema

The schema for the collections is defined in the `_create_schema` method of the `WeaviateClient` class. You can modify this to add additional properties or collections as needed.

## Troubleshooting

- **Connection Issues**: Ensure your Weaviate cluster URL and API key are correct
- **Missing Embeddings**: Check that your embedding API key is set correctly
- **Empty Results**: Make sure your ETL pipeline has processed the data before searching