# ETL Pipeline with Embeddings and Vector Storage

This extension to the AI Agent ETL Pipeline adds support for generating embeddings and storing them in a Weaviate vector database.

## Features

- Processes text files and extracts structured elements
- Copies image files to maintain a complete content repository
- Generates embeddings for text elements using sentence-transformers
- Stores elements and their embeddings in a Weaviate vector database
- Provides a query interface for semantic search

## Setup

### Prerequisites

- Docker and Docker Compose
- Python 3.8+ (for local execution)
- Required Python packages:
  - weaviate-client
  - sentence-transformers
  - requests

### Docker Execution (Recommended)

1. Run the complete stack with Docker Compose:

```sh
docker-compose -f docker-compose-weaviate.yml up
```

This will:
- Start Weaviate vector database
- Run the ETL process to extract text, generate embeddings, and store in Weaviate

2. Wait for the ETL process to complete (you'll see a summary in the logs)

### Local Execution

If you prefer to run the ETL process locally:

1. Start the Weaviate vector database:

```sh
docker-compose -f docker-compose-weaviate.yml up weaviate
```

2. Install required packages:

```sh
pip install weaviate-client sentence-transformers requests
```

3. Run the ETL script:

```sh
python etl_with_embeddings.py
```

## Querying the Vector Database

Once data is loaded into Weaviate, you can query it using semantic search:

```sh
python query_weaviate.py "your search query here"
```

This will:
1. Convert your query to an embedding
2. Find the most semantically similar content in the vector database
3. Display the top results

Example:

```sh
python query_weaviate.py "How to configure a camera"
```

## Customization

### Embedding Model

The default embedding model is `all-MiniLM-L6-v2`. You can change it by modifying the `load_embedding_model` function in `etl_with_embeddings.py`.

### Weaviate Schema

The Weaviate schema is defined in the `setup_weaviate_schema` function. You can modify it to add additional properties or classes.

### Processing Options

You can customize processing options in the `main` function:
- Set `use_weaviate = False` to disable Weaviate integration
- Modify input and output directories as needed
- Change the recursive option to control directory traversal

## Architecture

The system follows this processing flow:

1. Extract: Read text files and extract structured elements
2. Transform: 
   - Generate embeddings for each text element
   - Create metadata for each element
3. Load: 
   - Save structured data as JSON files
   - Copy image files to output directory
   - Store elements and embeddings in Weaviate vector database

## Notes

- The first run will download the embedding model, which may take some time
- Processing large document collections may be memory-intensive
- Make sure Weaviate is running before attempting to query the database