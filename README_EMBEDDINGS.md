# ETL Pipeline with Text & Image Embeddings and Vector Storage

This extension to the AI Agent ETL Pipeline adds support for generating embeddings for both text and images, storing them in a Weaviate vector database.

## Features

- Processes text files and extracts structured elements
- Applies contextual chunking to preserve semantic context
- Copies image files to maintain a complete content repository
- Generates embeddings for:
  - Text elements using sentence-transformers
  - Images using CLIP vision-language model
- Stores elements and their embeddings in a Weaviate vector database
- Provides a powerful query interface for:
  - Semantic text search with context-aware chunk recombination
  - Finding images by text description
  - Finding similar images

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

Once data is loaded into Weaviate, you can query it using different methods:

### Text Search

Find text content semantically related to your query:

```sh
python query_weaviate.py --text "your search query here"
```

Example:
```sh
python query_weaviate.py --text "How to configure a camera"
```

By default, chunks from the same document are grouped and merged to provide more complete context. To disable this and see individual chunks:

```sh
python query_weaviate.py --text --no-group "your search query here"
```

### Image Search by Text Description

Find images based on a text description:

```sh
python query_weaviate.py --image-by-text "description of image"
```

Example:
```sh
python query_weaviate.py --image-by-text "drone with camera"
```

### Similar Image Search

Find images similar to a reference image:

```sh
python query_weaviate.py --similar-to /path/to/your/image.jpg
```

Example:
```sh
python query_weaviate.py --similar-to processed_data/images/documentation/images/voxl2/m0054-hero-f.png
```

### Combined Search

If you don't specify a search type, both text and image search will be performed:

```sh
python query_weaviate.py "drone camera setup"
```

This will:
1. Convert your query to embeddings
2. Find semantically similar text content (with chunk grouping)
3. Find images related to your query
4. Display both sets of results

## Customization

### Embedding Models

Two different embedding models are used:

- **Text Embedding Model**: The default is `all-MiniLM-L6-v2`, a compact but powerful text embedding model.
  - You can change it by modifying the `load_text_embedding_model` function in `etl_with_embeddings.py`
  
- **Image Embedding Model**: The default is `clip-ViT-B-32`, a vision-language model that can generate embeddings for images.
  - You can change it by modifying the `load_image_embedding_model` function in `etl_with_embeddings.py`
  - Other options include `clip-ViT-B-16` (higher quality but slower) or `clip-ViT-L-14` (highest quality, much slower)

### Contextual Chunking

Contextual chunking is enabled by default with these settings:

- **Chunk Size**: 1000 characters per chunk
- **Chunk Overlap**: 200 characters of overlap between chunks

You can customize these settings in the `main` function of `etl_with_embeddings.py`:

```python
# Configuration options
use_weaviate = True    # Set to False to disable Weaviate integration
use_chunking = True    # Set to True to enable contextual chunking
chunk_size = 1000      # Maximum size of each chunk in characters
chunk_overlap = 200    # Overlap between consecutive chunks
```

Chunking creates overlapping segments of text that:
- Maintain context across document boundaries
- Allow for more precise vector representation
- Break down large documents into manageable pieces
- Improve retrieval of longer passages

### Weaviate Schema

The Weaviate schema is defined in the `setup_weaviate_schema` function. You can modify it to add additional properties or classes.

### Processing Options

You can customize processing options in the `main` function:
- Set `use_weaviate = False` to disable Weaviate integration
- Modify input and output directories as needed
- Change the recursive option to control directory traversal

## Architecture

The system follows this processing flow:

1. Extract: 
   - Read text files and extract structured elements
   - Apply contextual chunking to create overlapping segments
   - Identify image files for processing

2. Transform: 
   - For text elements and chunks:
     - Generate text embeddings using sentence-transformers
     - Create metadata for each element (including chunk position)
   - For images:
     - Generate image embeddings using CLIP
     - Create metadata including file path and image ID

3. Load: 
   - Save structured text data as JSON files
   - Copy image files to output directory
   - Store in Weaviate vector database:
     - Text elements in the `DocumentElement` class
     - Images in the `ImageElement` class

4. Query:
   - Text search with chunk recombination
   - Image search via text description
   - Image similarity search

The contextual chunking algorithm:
1. Splits documents into overlapping chunks
2. Tries to break at natural boundaries (paragraphs, sentences)
3. Maintains context through controlled overlap
4. Optimizes retrieval by balancing chunk size and specificity

## Notes

- The first run will download the embedding model, which may take some time
- Processing large document collections may be memory-intensive
- Make sure Weaviate is running before attempting to query the database