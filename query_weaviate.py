#!/usr/bin/env python3
"""
Script to query the Weaviate database for semantically similar content.
"""
import weaviate
import sys
from sentence_transformers import SentenceTransformer

def connect_to_weaviate(url="http://localhost:8080"):
    """Connect to Weaviate."""
    try:
        client = weaviate.Client(url=url)
        return client
    except Exception as e:
        print(f"Error connecting to Weaviate: {str(e)}")
        return None

def get_embedding(text, model_name="all-MiniLM-L6-v2"):
    """Generate an embedding for the query text."""
    model = SentenceTransformer(model_name)
    return model.encode(text).tolist()

def semantic_search(query, client, limit=5):
    """Perform a semantic search in Weaviate."""
    if client is None:
        print("No Weaviate connection.")
        return []
    
    # Generate embedding for the query
    query_embedding = get_embedding(query)
    
    # Search Weaviate
    try:
        result = client.query.get(
            "DocumentElement", 
            ["text", "type", "source_file", "element_id"]
        ).with_near_vector({
            "vector": query_embedding
        }).with_limit(limit).do()
        
        if "data" in result and "Get" in result["data"] and "DocumentElement" in result["data"]["Get"]:
            return result["data"]["Get"]["DocumentElement"]
        return []
    except Exception as e:
        print(f"Error searching Weaviate: {str(e)}")
        return []

def display_results(results):
    """Display search results."""
    if not results:
        print("No results found.")
        return
    
    print(f"\nFound {len(results)} results:")
    print("-" * 80)
    
    for i, item in enumerate(results):
        print(f"Result {i+1}:")
        print(f"Type: {item['type']}")
        print(f"Source: {item['source_file']}")
        print(f"Text: {item['text'][:200]}..." if len(item['text']) > 200 else f"Text: {item['text']}")
        print("-" * 80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python query_weaviate.py 'your search query'")
        return
    
    # Get query from command line argument
    query = " ".join(sys.argv[1:])
    print(f"Searching for: '{query}'")
    
    # Connect to Weaviate
    client = connect_to_weaviate()
    if not client:
        print("Could not connect to Weaviate. Is it running?")
        return
    
    # Perform search
    results = semantic_search(query, client, limit=5)
    
    # Display results
    display_results(results)

if __name__ == "__main__":
    main()