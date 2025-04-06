# AI Agent ETL Pipeline with Weaviate Multimodal Integration

This extension of the AI Agent ETL Pipeline adds Weaviate vector database integration for efficient multimodal semantic search across both text and image content.

## Features

- Store text and image embeddings in Weaviate vector database
- Multimodal search capabilities (text-to-text, text-to-image, image-to-text, image-to-image)
- Support for both local Weaviate instance and Weaviate Cloud Service
- Multiple embedding provider options (OpenAI, Cohere)
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

The `query_with_weaviate.py` script supports several command-line arguments for both text and image searches:

```bash
# Text search
python query_with_weaviate.py "your search query" --top-k 10

# Image search
python query_with_weaviate.py "optional context" --image path/to/image.jpg

# Full options
python query_with_weaviate.py "your search query" --top-k 10 --cluster-url https://your-cluster-url.weaviate.cloud --api-key your-api-key --embedding-api-key your-embedding-api-key
```

- `query`: The search query text (required)
- `--image`: Path to an image file for image-based search (optional)
- `--top-k`: Number of top results to return (default: 5)
- `--cluster-url`: URL of Weaviate Cloud cluster
- `--api-key`: API key for Weaviate Cloud
- `--embedding-api-key`: API key for embedding provider (OpenAI or Cohere)

## Development and Customization

### Multimodal Embedding Models

This implementation supports several embedding options for both text and images:

1. **Text Embeddings**:
   - OpenAI (`text-embedding-3-small`) when `OPENAI_API_KEY` is provided
   - Cohere (`embed-english-v3.0`) when `COHERE_API_KEY` is provided
   - Mock embeddings as fallback

2. **Image Embeddings**:
   - Uses text-based context models with the image filename
   - Stores the actual image data for use with Weaviate's CLIP integration
   - Falls back to mock embeddings when needed

The implementation uses a dedicated module (`multimodal_embeddings.py`) for generating embeddings:

```python
# Text embedding function in multimodal_embeddings.py
def get_text_embedding(text: str) -> List[float]:
    # Try OpenAI first
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if openai_api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_api_key)
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"OpenAI embedding failed: {str(e)}")
    
    # Try Cohere as fallback
    cohere_api_key = os.environ.get("COHERE_API_KEY")
    if cohere_api_key:
        try:
            import cohere
            co = cohere.Client(cohere_api_key)
            response = co.embed(
                texts=[text],
                model="embed-english-v3.0" 
            )
            return response.embeddings[0]
        except Exception as e:
            logger.warning(f"Cohere embedding failed: {str(e)}")
    
    # Use mock embeddings as final fallback
    return get_mock_embedding(text)
```

### Native Multimodal Search

The system supports Weaviate's native multimodal CLIP capabilities:

- Uses `multi2vec_clip` module for multimodal operations
- Supports both vector-based and native search methods
- Adaptively falls back to vector search when native methods aren't available
- Stores base64-encoded image data for proper multimodal processing

### Collection Schema

The schema for the collections is defined in the `_create_schema` method of the `WeaviateClient` class. You can modify this to add additional properties or collections as needed.

## Troubleshooting

- **Connection Issues**: Ensure your Weaviate cluster URL and API key are correct
- **Missing Embeddings**: Check that your embedding API key is set correctly
- **Empty Results**: Make sure your ETL pipeline has processed the data before searching