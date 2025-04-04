#!/usr/bin/env python3
"""
Query script that uses Weaviate for semantic search.
This script demonstrates how to perform semantic search queries using Weaviate.
"""
import os
import argparse
import time
from typing import List, Dict, Any
from bulk_process_files_with_phoenix import get_mock_embedding
from weaviate_client import WeaviateClient

def semantic_search(
    query: str, 
    weaviate_client: WeaviateClient,
    top_k: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Perform semantic search on the embeddings using Weaviate.
    
    Args:
        query: The search query
        weaviate_client: The Weaviate client
        top_k: Number of top results to return
        
    Returns:
        Dictionary with search results from different collections
    """
    print(f"\nPerforming semantic search for: '{query}'")
    print("Generating embedding for query...")
    
    # Generate embedding for the query
    search_start_time = time.time()
    embedding_start_time = time.time()
    query_embedding = get_mock_embedding(query)
    embedding_duration = (time.time() - embedding_start_time) * 1000
    
    print(f"Embedding generated in {embedding_duration:.2f}ms")
    
    # Perform the search across all collections
    results = weaviate_client.search(query_embedding=query_embedding, limit=top_k)
    
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
        combined_results.append({
            "type": "text",
            "display_text": result.properties.get("text", "")[:100] + "..." if len(result.properties.get("text", "")) > 100 else result.properties.get("text", ""),
            "element_type": result.properties.get("type", "Text"),
            "source": os.path.basename(result.properties.get("source_file", "Unknown")),
            "certainty": result.metadata.certainty if hasattr(result, "metadata") and hasattr(result.metadata, "certainty") else None
        })
    
    # Format image results
    for result in image_results:
        combined_results.append({
            "type": "image",
            "display_text": f"[Image: {result.properties.get('file_name', 'unknown')}]",
            "element_type": "Image",
            "source": os.path.basename(result.properties.get("image_path", "Unknown")),
            "certainty": result.metadata.certainty if hasattr(result, "metadata") and hasattr(result.metadata, "certainty") else None
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
    parser = argparse.ArgumentParser(description="Semantic search with Weaviate")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--cluster-url", help="URL of Weaviate Cloud cluster")
    parser.add_argument("--api-key", help="API key for Weaviate Cloud")
    parser.add_argument("--embedding-api-key", help="API key for embedding provider (OpenAI or Cohere)")
    args = parser.parse_args()
    
    try:
        # Connect to Weaviate
        weaviate_client = WeaviateClient(
            cluster_url=args.cluster_url,
            api_key=args.api_key,
            embedding_api_key=args.embedding_api_key
        )
        
        # Perform search
        results = semantic_search(
            query=args.query,
            weaviate_client=weaviate_client,
            top_k=args.top_k
        )
        
        # Format and display results
        format_results(results)
        
    except Exception as e:
        print(f"Error: {e}")
        
        # Show usage instructions if we can't connect
        if "Weaviate URL and API key must be provided" in str(e):
            print("\nUsage examples:")
            print("  With environment variables:")
            print("    export WEAVIATE_URL=https://your-cluster-url.weaviate.cloud")
            print("    export WEAVIATE_API_KEY=your-api-key")
            print("    export OPENAI_API_KEY=your-openai-api-key")
            print("    python query_with_weaviate.py 'your search query'")
            print("\n  With command line arguments:")
            print("    python query_with_weaviate.py 'your search query' --cluster-url https://your-cluster-url.weaviate.cloud --api-key your-api-key --embedding-api-key your-openai-api-key")

if __name__ == "__main__":
    main()