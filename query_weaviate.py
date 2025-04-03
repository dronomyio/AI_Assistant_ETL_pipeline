#!/usr/bin/env python3
"""
Script to query the Weaviate database for semantically similar content.
"""
import weaviate
import sys
import os
from PIL import Image
from sentence_transformers import SentenceTransformer

def connect_to_weaviate(url="http://localhost:8080"):
    """Connect to Weaviate."""
    try:
        client = weaviate.Client(url=url)
        return client
    except Exception as e:
        print(f"Error connecting to Weaviate: {str(e)}")
        return None

def get_text_embedding(text, model_name="all-MiniLM-L6-v2"):
    """Generate an embedding for the query text."""
    model = SentenceTransformer(model_name)
    return model.encode(text).tolist()

def get_image_embedding(image_path, model_name="clip-ViT-B-32"):
    """Generate an embedding for the image."""
    model = SentenceTransformer(model_name)
    img = Image.open(image_path)
    return model.encode(img).tolist()

def semantic_text_search(query, client, limit=5, group_chunks=True):
    """
    Perform a semantic search for text in Weaviate.
    
    Args:
        query: The search query
        client: Weaviate client
        limit: Maximum number of results to return
        group_chunks: If True, group chunks from the same document together
    """
    if client is None:
        print("No Weaviate connection.")
        return []
    
    # Generate embedding for the query
    query_embedding = get_text_embedding(query)
    
    # Search Weaviate for similar text
    try:
        # Get more results if we're grouping chunks
        actual_limit = limit * 3 if group_chunks else limit
        
        result = client.query.get(
            "DocumentElement", 
            ["text", "type", "source_file", "element_id", "is_chunk", "chunk_index", "total_chunks", "metadata"]
        ).with_near_vector({
            "vector": query_embedding
        }).with_limit(actual_limit).do()
        
        if "data" in result and "Get" in result["data"] and "DocumentElement" in result["data"]["Get"]:
            results = result["data"]["Get"]["DocumentElement"]
            
            # Process results if we need to group chunks
            if group_chunks:
                # Group chunks by source file
                grouped_results = {}
                non_chunks = []
                
                for item in results:
                    # Skip non-relevant fields
                    is_chunk = item.get("is_chunk", False)
                    
                    if is_chunk:
                        source = item["source_file"]
                        if source not in grouped_results:
                            grouped_results[source] = []
                        grouped_results[source].append(item)
                    else:
                        non_chunks.append(item)
                
                # Sort chunks by index
                for source in grouped_results:
                    grouped_results[source].sort(key=lambda x: x.get("chunk_index", 0))
                
                # Merge chunks from the same source and add them to results
                final_results = []
                
                # First add non-chunks
                final_results.extend(non_chunks)
                
                # Then add merged chunks (up to limit)
                for source in grouped_results:
                    chunks = grouped_results[source]
                    if not chunks:
                        continue
                    
                    # Merge chunks
                    merged_text = " ".join([chunk["text"] for chunk in chunks])
                    chunk_ids = [chunk["element_id"] for chunk in chunks]
                    
                    # Create merged result
                    merged_result = {
                        "text": merged_text,
                        "type": "MergedChunks",
                        "source_file": source,
                        "element_id": chunk_ids[0],  # Use first chunk's ID
                        "metadata": {
                            "merged_from": chunk_ids,
                            "chunk_count": len(chunks)
                        }
                    }
                    
                    final_results.append(merged_result)
                
                # Limit final results
                return final_results[:limit]
            
            return results[:limit]
        return []
    except Exception as e:
        print(f"Error searching Weaviate for text: {str(e)}")
        return []

def semantic_image_search(query, client, limit=5):
    """Perform a semantic search for images related to a text query."""
    if client is None:
        print("No Weaviate connection.")
        return []
    
    # Generate embedding for the query
    query_embedding = get_text_embedding(query)
    
    # Search Weaviate for similar images
    try:
        result = client.query.get(
            "ImageElement", 
            ["file_path", "file_name", "source_file", "image_id"]
        ).with_near_vector({
            "vector": query_embedding
        }).with_limit(limit).do()
        
        if "data" in result and "Get" in result["data"] and "ImageElement" in result["data"]["Get"]:
            return result["data"]["Get"]["ImageElement"]
        return []
    except Exception as e:
        print(f"Error searching Weaviate for images: {str(e)}")
        return []

def find_similar_image(image_path, client, limit=5):
    """Find images similar to a given image."""
    if client is None:
        print("No Weaviate connection.")
        return []
    
    # Generate embedding for the image
    query_embedding = get_image_embedding(image_path)
    
    # Search Weaviate for similar images
    try:
        result = client.query.get(
            "ImageElement", 
            ["file_path", "file_name", "source_file", "image_id"]
        ).with_near_vector({
            "vector": query_embedding
        }).with_limit(limit).do()
        
        if "data" in result and "Get" in result["data"] and "ImageElement" in result["data"]["Get"]:
            return result["data"]["Get"]["ImageElement"]
        return []
    except Exception as e:
        print(f"Error searching Weaviate for similar images: {str(e)}")
        return []

def display_text_results(results):
    """Display text search results."""
    if not results:
        print("No text results found.")
        return
    
    print(f"\nFound {len(results)} text results:")
    print("-" * 80)
    
    for i, item in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Type: {item['type']}")
        print(f"Source: {item['source_file']}")
        
        # Handle merged chunks differently
        if item['type'] == "MergedChunks" and "metadata" in item and "chunk_count" in item["metadata"]:
            chunk_count = item["metadata"]["chunk_count"]
            print(f"[Merged from {chunk_count} chunks]")
            
        # Truncate long text for display
        if len(item['text']) > 200:
            print(f"Text: {item['text'][:200]}...")
        else:
            print(f"Text: {item['text']}")
            
        print("-" * 80)

def display_image_results(results):
    """Display image search results."""
    if not results:
        print("No image results found.")
        return
    
    print(f"\nFound {len(results)} image results:")
    print("-" * 80)
    
    for i, item in enumerate(results):
        print(f"Image Result {i+1}:")
        print(f"File: {item['file_name']}")
        print(f"Source: {item['source_file']}")
        print(f"Path: {item['file_path']}")
        print("-" * 80)

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Text search: python query_weaviate.py --text 'your search query'")
        print("  Text search (no chunk grouping): python query_weaviate.py --text --no-group 'your query'")
        print("  Image search by text: python query_weaviate.py --image-by-text 'description of image'")
        print("  Similar image search: python query_weaviate.py --similar-to /path/to/image.jpg")
        print("  Combined search: python query_weaviate.py 'your search query'")
        return
    
    # Connect to Weaviate
    client = connect_to_weaviate()
    if not client:
        print("Could not connect to Weaviate. Is it running?")
        return
    
    # Parse arguments
    if sys.argv[1] == "--text":
        # Check for --no-group flag
        group_chunks = True
        start_idx = 2
        
        if len(sys.argv) > 2 and sys.argv[2] == "--no-group":
            group_chunks = False
            start_idx = 3
            
        if len(sys.argv) <= start_idx:
            print("Error: No search query provided")
            return
            
        # Text search
        query = " ".join(sys.argv[start_idx:])
        print(f"Searching for text related to: '{query}'")
        print(f"Chunk grouping: {'Enabled' if group_chunks else 'Disabled'}")
        
        results = semantic_text_search(query, client, limit=5, group_chunks=group_chunks)
        display_text_results(results)
        
    elif sys.argv[1] == "--image-by-text" and len(sys.argv) > 2:
        # Image search by text description
        query = " ".join(sys.argv[2:])
        print(f"Searching for images related to: '{query}'")
        results = semantic_image_search(query, client, limit=5)
        display_image_results(results)
        
    elif sys.argv[1] == "--similar-to" and len(sys.argv) > 2:
        # Find similar images
        image_path = sys.argv[2]
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return
            
        print(f"Finding images similar to: {image_path}")
        results = find_similar_image(image_path, client, limit=5)
        display_image_results(results)
        
    else:
        # Default to combined search if no flag is provided
        query = " ".join(sys.argv[1:])
        print(f"Searching for: '{query}'")
        
        # Do both text and image search
        text_results = semantic_text_search(query, client, limit=5, group_chunks=True)
        image_results = semantic_image_search(query, client, limit=5)
        
        # Display results
        display_text_results(text_results)
        display_image_results(image_results)

if __name__ == "__main__":
    main()