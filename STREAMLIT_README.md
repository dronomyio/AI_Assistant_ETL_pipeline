# AI ETL Pipeline Visualizer

A Streamlit app for visualizing and interacting with the AI ETL Pipeline with vector embeddings.

## Features

- Process text with contextual chunking
- Generate and visualize embeddings for text and images
- Explore vector embeddings through interactive visualizations
- Perform semantic searches with different options:
  - Text-to-text search with chunk grouping
  - Text-to-image search
  - Image-to-image similarity search
  - Combined search

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements_streamlit.txt
```

2. Make sure Weaviate is running:

```bash
docker-compose -f docker-compose-weaviate.yml up -d
```

## Usage

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

## App Structure

The app is organized into four main tabs:

1. **Process Text** - Process text documents with configurable contextual chunking
2. **Process Images** - Upload and generate embeddings for images
3. **Explore Embeddings** - Visualize embeddings with PCA or UMAP projections
4. **Search** - Perform various types of semantic searches

## Configuration

In the sidebar, you can:
- Connect to your Weaviate instance
- Configure chunking parameters
- Load the embedding model
- See statistics about stored data

## Search Options

The app supports multiple search modes:

- **Text Search** - Find text content semantically related to your query
  - Option to group chunks from the same document
  - Adjustable result limit

- **Image Search** - Two methods:
  - By Text Description - Find images matching a text description
  - By Similarity - Upload an image to find similar ones

- **Combined Search** - Search for both text and images with a single query

## Visualizations

The app provides interactive visualizations:
- Embedding value distributions for images
- 2D projections of embeddings using PCA or UMAP
- Color-coded embedding clusters by element type

## Requirements

- Python 3.8+
- Streamlit
- Weaviate (running instance)
- Sentence Transformers
- Various data science libraries (numpy, pandas, etc.)