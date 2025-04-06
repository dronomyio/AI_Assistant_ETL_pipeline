#!/usr/bin/env python3
"""
Query script that uses Weaviate for semantic search.
This script demonstrates how to perform semantic search queries using Weaviate.
"""
import os
import argparse
import time
import weaviate
from weaviate.classes.init import Auth
from typing import List, Dict, Any, Optional
import base64
from pathlib import Path

# Import multimodal embedding functions
from multimodal_embeddings import get_text_embedding, get_image_embedding, image_to_base64

def connect_to_weaviate(
    cluster_url: Optional[str] = None,
    api_key: Optional[str] = None,
    embedding_api_key: Optional[str] = None
) -> weaviate.WeaviateClient:
    """
    Connect to Weaviate client.
    
    Args:
        cluster_url: URL of the Weaviate cluster
        api_key: API key for Weaviate Cloud
        embedding_api_key: API key for embedding provider (OpenAI or Cohere)
        
    Returns:
        Weaviate client
    """
    # Get credentials from environment variables if not provided
    cluster_url = cluster_url or os.environ.get("WEAVIATE_URL")
    api_key = api_key or os.environ.get("WEAVIATE_API_KEY")
    embedding_api_key = embedding_api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("COHERE_API_KEY")
    
    if not cluster_url or not api_key:
        raise ValueError("Weaviate URL and API key must be provided or set as environment variables")
    
    # Set up headers for embedding provider
    headers = {}
    if embedding_api_key:
        # Determine which embedding provider to use based on environment variables
        if os.environ.get("OPENAI_API_KEY"):
            headers["X-OpenAI-Api-Key"] = embedding_api_key
        elif os.environ.get("COHERE_API_KEY"):
            headers["X-Cohere-Api-Key"] = embedding_api_key
    
    # Connect to Weaviate Cloud
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=cluster_url,
        auth_credentials=Auth.api_key(api_key),
        headers=headers
    )
    
    # Check if connection is ready
    if client.is_ready():
        print(f"Successfully connected to Weaviate Cloud at {cluster_url}")
    else:
        raise ConnectionError(f"Failed to connect to Weaviate Cloud at {cluster_url}")
    
    return client

def semantic_search(
    query: str, 
    weaviate_client: weaviate.WeaviateClient,
    image_path: Optional[str] = None,
    top_k: int = 5
) -> Dict[str, List[Any]]:
    """
    Perform semantic search on the embeddings using Weaviate.
    
    Args:
        query: The search query (text)
        weaviate_client: The Weaviate client
        image_path: Optional path to an image file for multimodal search
        top_k: Number of top results to return
        
    Returns:
        Dictionary with search results from different collections
    """
    search_start_time = time.time()
    results = {}
    collections = ["TextChunk", "ImageEmbedding"]
    
    # Check if we're doing an image-based search
    if image_path and os.path.exists(image_path):
        print(f"\nPerforming image-based search using: '{image_path}'")
        
        # Generate embedding for the image
        embedding_start_time = time.time()
        image_embedding = get_image_embedding(image_path)
        embedding_duration = (time.time() - embedding_start_time) * 1000
        print(f"Image embedding generated in {embedding_duration:.2f}ms")
        
        # Convert image to base64 for native multimodal search
        try:
            image_base64 = image_to_base64(image_path)
            
            # Search in collections using vector-based approach
            for collection_name in collections:
                try:
                    collection = weaviate_client.collections.get(collection_name)
                    
                    # Try using native near_image if possible, otherwise fall back to vector search
                    try:
                        query_result = collection.query.near_image(
                            near_image=image_base64,
                            limit=top_k
                        )
                        results[collection_name] = query_result.objects
                        print(f"Found {len(query_result.objects)} results in {collection_name} using native image search")
                    except Exception:
                        # Fall back to vector search
                        query_result = collection.query.near_vector(
                            near_vector=image_embedding,
                            limit=top_k
                        )
                        results[collection_name] = query_result.objects
                        print(f"Found {len(query_result.objects)} results in {collection_name} using vector search")
                        
                except Exception as e:
                    print(f"Error searching collection {collection_name}: {e}")
                    results[collection_name] = []
            
        except Exception as e:
            print(f"Error with image-based search: {e}")
            print("Falling back to text search...")
            # If image processing fails, fall back to text search
            
    # If no image provided or image search failed, do text-based search
    if not image_path or not results.get("TextChunk"):
        print(f"\nPerforming text-based search for: '{query}'")
        print("Generating embedding for query...")
        
        # Generate embedding for the query
        embedding_start_time = time.time()
        query_embedding = get_text_embedding(query)
        embedding_duration = (time.time() - embedding_start_time) * 1000
        print(f"Text embedding generated in {embedding_duration:.2f}ms")
        
        # Search in collections using the text embedding
        for collection_name in collections:
            try:
                collection = weaviate_client.collections.get(collection_name)
                
                # Try using native near_text if possible, otherwise fall back to vector search
                try:
                    query_result = collection.query.near_text(
                        query=query,
                        limit=top_k
                    )
                    results[collection_name] = query_result.objects
                    print(f"Found {len(query_result.objects)} results in {collection_name} using native text search")
                except Exception:
                    # Fall back to vector search
                    query_result = collection.query.near_vector(
                        near_vector=query_embedding,
                        limit=top_k
                    )
                    results[collection_name] = query_result.objects
                    print(f"Found {len(query_result.objects)} results in {collection_name} using vector search")
                    
            except Exception as e:
                print(f"Error searching collection {collection_name}: {e}")
                results[collection_name] = []
    
    search_duration = (time.time() - search_start_time) * 1000
    print(f"Search completed in {search_duration:.2f}ms")
    
    return results

def format_results(results: Dict[str, List[Dict[str, Any]]]) -> None:
    """
    Format and print search results.
    
    Args:
        results: Dictionary with search results from different collections
    """
    text_results = results.get("TextChunk", [])
    image_results = results.get("ImageEmbedding", [])
    
    combined_results = []
    
    # Format text results
    for result in text_results:
        # Extract properties safely
        properties = getattr(result, "properties", {})
        if not properties:
            properties = {}
            if hasattr(result, "get"):
                properties = result
        
        # Extract text and handle property access
        text = ""
        if hasattr(properties, "get"):
            text = properties.get("text", "")
        elif hasattr(properties, "text"):
            text = getattr(properties, "text", "")
        
        # Format text for display
        display_text = text[:100] + "..." if len(text) > 100 else text
        
        # Get element type
        element_type = "Text"
        if hasattr(properties, "get"):
            element_type = properties.get("type", "Text")
        elif hasattr(properties, "type"):
            element_type = getattr(properties, "type", "Text")
        
        # Get source file
        source_file = "Unknown"
        if hasattr(properties, "get"):
            source_file = properties.get("source_file", "Unknown")
        elif hasattr(properties, "source_file"):
            source_file = getattr(properties, "source_file", "Unknown")
        
        # Extract similarity/certainty score
        certainty = None
        if hasattr(result, "metadata") and hasattr(result.metadata, "certainty"):
            certainty = result.metadata.certainty
        elif hasattr(result, "distance"):
            certainty = 1.0 - getattr(result, "distance", 0) 
            
        combined_results.append({
            "type": "text",
            "display_text": display_text,
            "element_type": element_type,
            "source": os.path.basename(source_file),
            "certainty": certainty
        })
    
    # Format image results
    for result in image_results:
        # Extract properties safely
        properties = getattr(result, "properties", {})
        if not properties:
            properties = {}
            if hasattr(result, "get"):
                properties = result
        
        # Get filename
        file_name = "unknown"
        if hasattr(properties, "get"):
            file_name = properties.get("file_name", "unknown")
        elif hasattr(properties, "file_name"):
            file_name = getattr(properties, "file_name", "unknown")
        
        # Get image path
        image_path = "Unknown"
        if hasattr(properties, "get"):
            image_path = properties.get("image_path", "Unknown")
        elif hasattr(properties, "image_path"):
            image_path = getattr(properties, "image_path", "Unknown")
        
        # Extract similarity/certainty score
        certainty = None
        if hasattr(result, "metadata") and hasattr(result.metadata, "certainty"):
            certainty = result.metadata.certainty
        elif hasattr(result, "distance"):
            certainty = 1.0 - getattr(result, "distance", 0)
            
        combined_results.append({
            "type": "image",
            "display_text": f"[Image: {file_name}]",
            "element_type": "Image",
            "source": os.path.basename(image_path),
            "certainty": certainty
        })
    
    # Sort by certainty if available
    if combined_results and "certainty" in combined_results[0] and combined_results[0]["certainty"] is not None:
        combined_results.sort(key=lambda x: x["certainty"] if x["certainty"] is not None else 0, reverse=True)
    
    # Display results
    print(f"\nTop {len(combined_results)} results:")
    for i, result in enumerate(combined_results):
        print(f"\n{i+1}. {result['element_type']}: {result['display_text']}")
        if result["certainty"] is not None:
            print(f"   Similarity: {result['certainty']:.4f}")
        print(f"   Source: {result['source']}")

def main():
    parser = argparse.ArgumentParser(description="Multimodal semantic search with Weaviate")
    parser.add_argument("query", help="Search query text (or dummy text if using --image)")
    parser.add_argument("--image", help="Path to image file for image-based search")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--cluster-url", help="URL of Weaviate Cloud cluster")
    parser.add_argument("--api-key", help="API key for Weaviate Cloud")
    parser.add_argument("--embedding-api-key", help="API key for embedding provider (OpenAI or Cohere)")
    args = parser.parse_args()
    
    try:
        # Connect to Weaviate
        weaviate_client = connect_to_weaviate(
            cluster_url=args.cluster_url,
            api_key=args.api_key,
            embedding_api_key=args.embedding_api_key
        )
        
        # Validate image path if provided
        image_path = args.image
        if image_path and not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            image_path = None
        
        # Perform search (multimodal if image provided)
        results = semantic_search(
            query=args.query,
            weaviate_client=weaviate_client,
            image_path=image_path,
            top_k=args.top_k
        )
        
        # Format and display results
        format_results(results)
        
        # Close the connection
        weaviate_client.close()
        
    except Exception as e:
        print(f"Error: {e}")
        
        # Show usage instructions if we can't connect
        if "Weaviate URL and API key must be provided" in str(e):
            print("\nUsage examples:")
            print("  Text search with environment variables:")
            print("    export WEAVIATE_URL=https://your-cluster-url.weaviate.cloud")
            print("    export WEAVIATE_API_KEY=your-api-key")
            print("    export OPENAI_API_KEY=your-openai-api-key")
            print("    python query_with_weaviate.py 'your search query'")
            print("\n  Image search example:")
            print("    python query_with_weaviate.py 'search context' --image path/to/image.jpg")
            print("\n  With command line arguments:")
            print("    python query_with_weaviate.py 'your search query' --cluster-url https://your-cluster-url.weaviate.cloud --api-key your-api-key --embedding-api-key your-openai-api-key")

if __name__ == "__main__":
    main()