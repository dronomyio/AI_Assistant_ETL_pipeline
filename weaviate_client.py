#!/usr/bin/env python3
"""
Weaviate client for storing and retrieving embeddings from the ETL pipeline.
Uses Weaviate client v4 API.
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


class WeaviateClient:
    """Client for storing and retrieving embeddings from Weaviate."""
    
    def __init__(
        self,
        cluster_url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_api_key: Optional[str] = None
    ):
        """
        Initialize the Weaviate client.
        
        Args:
            cluster_url: URL of the Weaviate cluster (e.g., from Weaviate Cloud Service)
            api_key: API key for Weaviate Cloud
            embedding_api_key: API key for embedding provider (e.g., Cohere, OpenAI)
        """
        # Get credentials from environment variables if not provided
        self.cluster_url = cluster_url or os.environ.get("WEAVIATE_URL")
        self.api_key = api_key or os.environ.get("WEAVIATE_API_KEY")
        self.embedding_api_key = embedding_api_key or os.environ.get("COHERE_API_KEY") or os.environ.get("OPENAI_API_KEY")
        
        if not self.cluster_url or not self.api_key:
            raise ValueError("Weaviate URL and API key must be provided or set as environment variables")
        
        # Set up headers for embedding provider
        headers = {}
        if self.embedding_api_key:
            # Determine which embedding provider to use based on environment variables
            if os.environ.get("COHERE_API_KEY"):
                headers["X-Cohere-Api-Key"] = self.embedding_api_key
            elif os.environ.get("OPENAI_API_KEY"):
                headers["X-OpenAI-Api-Key"] = self.embedding_api_key
        
        # Connect to Weaviate Cloud
        self.client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.cluster_url,
            auth_credentials=Auth.api_key(self.api_key),
            headers=headers
        )
        
        # Check if connection is ready
        if self.client.is_ready():
            print(f"Successfully connected to Weaviate Cloud at {self.cluster_url}")
        else:
            raise ConnectionError(f"Failed to connect to Weaviate Cloud at {self.cluster_url}")
        
        # Create schema if it doesn't exist
        self._create_schema()

    def _create_schema(self):
        """Create the schema for document chunks and image embeddings."""
        # Check if our collections exist first
        collections = self.client.collections.list_all()
        collection_names = [c.name for c in collections]
        
        # Create TextChunk collection if it doesn't exist
        if "TextChunk" not in collection_names:
            text_chunk = self.client.collections.create(
                name="TextChunk",
                description="A chunk of text from a document",
                properties=[
                    Property(
                        name="text", 
                        data_type=DataType.TEXT,
                        description="The text content of the chunk"
                    ),
                    Property(
                        name="type", 
                        data_type=DataType.TEXT,
                        description="The type of the text (Title, Text, etc.)"
                    ),
                    Property(
                        name="source_file", 
                        data_type=DataType.TEXT,
                        description="The source file path"
                    ),
                    Property(
                        name="element_id", 
                        data_type=DataType.TEXT,
                        description="Unique ID for the element"
                    )
                ],
                vectorizer_config=Configure.Vectorizer.none()
            )
            print("Created TextChunk collection")
        
        # Create ImageEmbedding collection if it doesn't exist
        if "ImageEmbedding" not in collection_names:
            image_embedding = self.client.collections.create(
                name="ImageEmbedding",
                description="Embedding for an image",
                properties=[
                    Property(
                        name="image_path", 
                        data_type=DataType.TEXT,
                        description="Path to the image"
                    ),
                    Property(
                        name="file_name", 
                        data_type=DataType.TEXT,
                        description="File name of the image"
                    )
                ],
                vectorizer_config=Configure.Vectorizer.none()
            )
            print("Created ImageEmbedding collection")
    
    def store_text_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Store text chunks with embeddings in Weaviate.
        
        Args:
            chunks: List of text chunks with embeddings
            batch_size: Number of objects to batch together in each request
        """
        # Get the collection
        collection = self.client.collections.get("TextChunk")
        
        # Use batch processing
        with collection.batch.dynamic() as batch:
            for i, chunk in enumerate(chunks):
                if i % batch_size == 0:
                    print(f"Processing chunks {i} to {min(i+batch_size, len(chunks))}")
                
                # Skip if no embedding or text
                if "embedding" not in chunk or "text" not in chunk:
                    continue
                
                # Extract properties
                properties = {
                    "text": chunk.get("text", ""),
                    "type": chunk.get("type", "Text"),
                    "source_file": chunk.get("source_file", ""),
                    "element_id": chunk.get("element_id", "")
                }
                
                # Add to batch with vector
                batch.add_object(
                    properties=properties,
                    vector=chunk["embedding"]
                )
                
        print(f"Stored {len(chunks)} text chunks in Weaviate")
    
    def store_image_embeddings(self, images: List[Dict[str, Any]], batch_size: int = 50):
        """
        Store image embeddings in Weaviate.
        
        Args:
            images: List of image paths with embeddings
            batch_size: Number of objects to batch together in each request
        """
        # Get the collection
        collection = self.client.collections.get("ImageEmbedding")
        
        # Use batch processing
        with collection.batch.dynamic() as batch:
            for i, image in enumerate(images):
                if i % batch_size == 0:
                    print(f"Processing images {i} to {min(i+batch_size, len(images))}")
                
                # Skip if no embedding or path
                if "embedding" not in image or "image_path" not in image:
                    continue
                
                # Add to batch with vector
                batch.add_object(
                    properties={
                        "image_path": image.get("image_path", ""),
                        "file_name": os.path.basename(image.get("image_path", ""))
                    },
                    vector=image["embedding"]
                )
                
        print(f"Stored {len(images)} image embeddings in Weaviate")
    
    def search(self, query_embedding: List[float], limit: int = 10, collection_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Search for similar objects using a query embedding.
        
        Args:
            query_embedding: The embedding vector to search with
            limit: Maximum number of results to return
            collection_names: List of collection names to search in (default: all)
            
        Returns:
            Dictionary with search results
        """
        collections = collection_names or ["TextChunk", "ImageEmbedding"]
        results = {}
        
        for collection_name in collections:
            try:
                collection = self.client.collections.get(collection_name)
                
                query_result = collection.query.near_vector(
                    near_vector=query_embedding,
                    limit=limit
                )
                
                results[collection_name] = query_result.objects
            except Exception as e:
                print(f"Error searching collection {collection_name}: {e}")
                results[collection_name] = []
                
        return results
    
    def import_from_processed_data(self, processed_dir: str):
        """
        Import processed data from the ETL pipeline into Weaviate.
        
        Args:
            processed_dir: Directory containing processed data
        """
        # Import text chunks
        text_chunks = []
        
        # Walk through the processed directory for JSON files
        json_files = list(Path(processed_dir).glob("**/*.json"))
        print(f"Found {len(json_files)} JSON files in {processed_dir}")
        
        for file_path in json_files:
            if "phoenix_traces" in str(file_path):
                continue
                
            try:
                with open(file_path, 'r') as f:
                    elements = json.load(f)
                    
                    for element in elements:
                        # Regenerate embeddings (since they're not stored in the JSON)
                        from bulk_process_files_with_phoenix import get_mock_embedding
                        element["embedding"] = get_mock_embedding(element["text"])
                        element["source_file"] = str(file_path)
                        text_chunks.append(element)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded {len(text_chunks)} text chunks from processed data")
        self.store_text_chunks(text_chunks)
        
        # Import image embeddings
        image_files = []
        for ext in ["jpg", "jpeg", "png", "gif", "svg", "webp"]:
            image_files.extend(list(Path(processed_dir).joinpath("images").glob(f"**/*.{ext}")))
        
        print(f"Found {len(image_files)} image files in {processed_dir}/images")
        
        image_embeddings = []
        for img_path in image_files:
            # Generate mock embedding for the image
            from bulk_process_files_with_phoenix import get_mock_embedding
            img_data = {
                "image_path": str(img_path),
                "embedding": get_mock_embedding(os.path.basename(img_path))
            }
            image_embeddings.append(img_data)
        
        self.store_image_embeddings(image_embeddings)
        print(f"Imported {len(image_embeddings)} image embeddings")


def main():
    """Command line interface for the Weaviate client."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Import embeddings into Weaviate")
    parser.add_argument("--processed-dir", default="processed_data", help="Directory containing processed data")
    parser.add_argument("--cluster-url", help="URL of Weaviate Cloud cluster")
    parser.add_argument("--api-key", help="API key for Weaviate Cloud")
    parser.add_argument("--embedding-api-key", help="API key for embedding provider (Cohere or OpenAI)")
    args = parser.parse_args()
    
    try:
        # Create .env file if arguments are provided and not already in environment
        if args.cluster_url or args.api_key or args.embedding_api_key:
            env_content = []
            
            if args.cluster_url and not os.environ.get("WEAVIATE_URL"):
                env_content.append(f"WEAVIATE_URL={args.cluster_url}")
                
            if args.api_key and not os.environ.get("WEAVIATE_API_KEY"):
                env_content.append(f"WEAVIATE_API_KEY={args.api_key}")
                
            if args.embedding_api_key:
                if not os.environ.get("COHERE_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
                    # Assume Cohere by default
                    env_content.append(f"COHERE_API_KEY={args.embedding_api_key}")
            
            # Write to .env file if we have any content
            if env_content:
                with open(".env", "w") as f:
                    f.write("\n".join(env_content))
                print("Created .env file with credentials")
                # Reload environment variables
                load_dotenv(override=True)
                
        # Connect to Weaviate
        client = WeaviateClient(
            cluster_url=args.cluster_url,
            api_key=args.api_key,
            embedding_api_key=args.embedding_api_key
        )
        
        # Import data
        client.import_from_processed_data(args.processed_dir)
        print("Data import complete!")
    
    except Exception as e:
        print(f"Error: {e}")
        

if __name__ == "__main__":
    main()