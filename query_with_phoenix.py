#!/usr/bin/env python3
"""
Query script with Phoenix observability for the ETL pipeline.
This script demonstrates how to perform semantic search queries with Phoenix monitoring.
"""
import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Import Phoenix observer
from phoenix_observer import PhoenixObserver

def load_processed_data(processed_dir: str) -> List[Dict[str, Any]]:
    """
    Load all processed elements from the processed data directory.
    
    Args:
        processed_dir: Directory containing processed data
        
    Returns:
        List of all elements with their metadata
    """
    all_elements = []
    
    # Walk through the processed directory
    for root, _, files in os.walk(processed_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        elements = json.load(f)
                        
                        # Add source file information
                        for element in elements:
                            element['source_file'] = file_path
                            
                        all_elements.extend(elements)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(all_elements)} elements from {processed_dir}")
    return all_elements

def mock_embedding_similarity(query_embedding: List[float], element_embedding: List[float]) -> float:
    """
    Mock function to calculate similarity between embeddings.
    In a real implementation, this would use cosine similarity or other metrics.
    
    Args:
        query_embedding: The query embedding vector
        element_embedding: The element embedding vector
        
    Returns:
        Similarity score (0-1)
    """
    import numpy as np
    
    # Generate a deterministic similarity based on text length
    # In a real implementation, this would be the actual cosine similarity
    return np.random.uniform(0.5, 0.95)

def get_mock_embedding(text: str) -> List[float]:
    """
    Generate a mock embedding for the text.
    In a real implementation, this would call an embedding model.
    
    Args:
        text: The text to generate an embedding for
        
    Returns:
        A list of floats representing the embedding
    """
    import hashlib
    import numpy as np
    
    # Create a deterministic but unique embedding based on the text
    text_hash = hashlib.md5(text.encode()).hexdigest()
    np.random.seed(int(text_hash[:8], 16))
    
    # Generate a mock embedding with 384 dimensions
    return np.random.normal(0, 1, 384).tolist()

def semantic_search(
    query: str, 
    elements: List[Dict[str, Any]], 
    top_k: int = 5,
    observer: PhoenixObserver = None
) -> List[Dict[str, Any]]:
    """
    Perform semantic search on the elements using the query.
    
    Args:
        query: The search query
        elements: List of elements to search through
        top_k: Number of top results to return
        observer: Phoenix observer for monitoring
        
    Returns:
        List of top k matching elements with similarity scores
    """
    trace_id = None
    search_start_time = time.time()
    
    if observer:
        trace_id = observer.start_trace(
            metadata={"query": query, "num_elements": len(elements)}
        )
    
    try:
        # In a real implementation, this would use a proper embedding model
        print(f"Generating embedding for query: '{query}'")
        
        # Log embedding generation
        embedding_start_time = time.time()
        query_embedding = get_mock_embedding(query)
        embedding_duration = (time.time() - embedding_start_time) * 1000
        
        if observer and trace_id:
            observer.log_embedding_span(
                trace_id=trace_id,
                embedding_model="all-MiniLM-L6-v2",
                text_or_image=query,
                embedding_type="text",
                embedding_dim=len(query_embedding),
                duration_ms=embedding_duration
            )
        
        # Calculate similarity scores
        results = []
        for element in elements:
            # In a real implementation, elements would have actual embeddings
            # Here we generate them on the fly for demonstration
            element_text = element.get('text', '')
            element_embedding = get_mock_embedding(element_text)
            
            # Calculate similarity (cosine similarity in a real implementation)
            similarity = mock_embedding_similarity(query_embedding, element_embedding)
            
            results.append({
                'element': element,
                'similarity': similarity,
                'text': element_text,
                'type': element.get('type', 'Unknown'),
                'source': element.get('source_file', 'Unknown')
            })
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Take top_k results
        top_results = results[:top_k]
        
        search_duration = (time.time() - search_start_time) * 1000
        
        # Log search to Phoenix
        if observer and trace_id:
            results_preview = [{
                'text': r['text'][:100] + '...' if len(r['text']) > 100 else r['text'],
                'type': r['type'],
                'similarity': r['similarity']
            } for r in top_results]
            
            observer.log_search_span(
                trace_id=trace_id,
                query=query,
                search_type="text",
                num_results=len(top_results),
                results_preview=results_preview,
                duration_ms=search_duration
            )
            
            # End trace
            observer.end_trace(trace_id, status="success")
        
        return top_results
        
    except Exception as e:
        print(f"Error performing search: {e}")
        
        # Log error to Phoenix
        if observer and trace_id:
            observer.end_trace(trace_id, status="error", error=str(e))
            
        return []

def main():
    parser = argparse.ArgumentParser(description="Semantic search with Phoenix observability")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--processed-dir", default="processed_data", help="Directory containing processed data")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--disable-phoenix", action="store_true", help="Disable Phoenix observability")
    args = parser.parse_args()
    
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up processed data path 
    processed_dir = args.processed_dir
    if not os.path.isabs(processed_dir):
        processed_dir = os.path.join(script_dir, processed_dir)
    
    # Initialize Phoenix observer
    observer = None
    if not args.disable_phoenix:
        try:
            observer = PhoenixObserver(
                enabled=True,
                app_name="ai-agent-etl-search",
                env="development"
            )
            print("Phoenix observability enabled")
        except Exception as e:
            print(f"Could not initialize Phoenix observer: {str(e)}")
    
    # Load all processed elements
    elements = load_processed_data(processed_dir)
    
    if not elements:
        print("No processed data found. Please run bulk_process_files_with_phoenix.py first.")
        return
    
    # Perform search
    print(f"\nPerforming semantic search for: '{args.query}'")
    results = semantic_search(
        query=args.query,
        elements=elements,
        top_k=args.top_k,
        observer=observer
    )
    
    # Display results
    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['type']}: {result['text'][:100]}..." if len(result['text']) > 100 else f"\n{i+1}. {result['type']}: {result['text']}")
        print(f"   Similarity: {result['similarity']:.4f}")
        print(f"   Source: {os.path.basename(result['source'])}")
    
    if observer:
        print("\nSearch telemetry sent to Phoenix")
        print("View results in Phoenix UI")

if __name__ == "__main__":
    main()