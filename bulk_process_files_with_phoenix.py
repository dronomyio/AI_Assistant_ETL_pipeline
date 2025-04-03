#!/usr/bin/env python3
"""
Enhanced script to process all text files with Phoenix observability.
This adds Arize Phoenix integration to the ETL pipeline for monitoring and debugging.
"""
import os
import json
import re
import shutil
import uuid
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

# Import Phoenix observer
from phoenix_observer import PhoenixObserver

def simple_text_processor(text):
    """
    A simple text processor that breaks text into elements.
    This is a simplified version of what the Unstructured API might do.
    """
    elements = []
    
    # Extract title (first line)
    lines = text.split('\n')
    if lines and lines[0].strip():
        elements.append({
            "type": "Title",
            "text": lines[0].strip(),
            "element_id": f"element-{uuid.uuid4()}"
        })
    
    # Process paragraphs (separated by blank lines)
    paragraphs = re.split(r'\n\s*\n', text)
    for i, para in enumerate(paragraphs):
        if para.strip():
            elements.append({
                "type": "Text",
                "text": para.strip(),
                "element_id": f"element-{i+1}"
            })
    
    return elements

def contextual_chunker(text, chunk_size=1000, chunk_overlap=200):
    """
    Split text into overlapping chunks to maintain context across chunks.
    
    Args:
        text: The text to split into chunks
        chunk_size: Maximum size of each chunk (in characters)
        chunk_overlap: Overlap between consecutive chunks (in characters)
    
    Returns:
        List of text chunks with context
    """
    # If text is smaller than chunk_size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Find the end of this chunk
        end = min(start + chunk_size, len(text))
        
        # If we're not at the end of the text,
        # try to break at a sentence or paragraph boundary
        if end < len(text):
            # First try to find a paragraph break
            last_break = text.rfind('\n\n', start, end)
            if last_break != -1 and last_break > start + chunk_size // 2:
                end = last_break
            else:
                # Try to find a sentence break (period followed by space)
                last_period = text.rfind('. ', start, end)
                if last_period != -1 and last_period > start + chunk_size // 2:
                    end = last_period + 1  # Include the period
        
        # Extract chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start to create overlap
        start = end - chunk_overlap if end - chunk_overlap > start else end
    
    return chunks

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

def process_file(
    file_path: str, 
    output_dir: str, 
    input_base_dir: str, 
    observer: Optional[PhoenixObserver] = None,
    trace_id: Optional[str] = None,
    use_chunking: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Tuple[int, int, bool]:
    """
    Process a single file and save the output to the specified directory.
    
    Args:
        file_path: Path to the file to process
        output_dir: Directory to save processed output to
        input_base_dir: Base directory of input files
        observer: Optional Phoenix observer for monitoring
        trace_id: Trace ID if using Phoenix
        use_chunking: Whether to use contextual chunking
        chunk_size: Size of chunks for chunking
        chunk_overlap: Overlap between chunks
        
    Returns:
        Tuple of (number of elements, number of embeddings, is_image)
    """
    try:
        # Create relative output path to maintain directory structure
        rel_path = os.path.relpath(file_path, input_base_dir)
        
        # Check if it's an image file
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp')):
            print(f"Copying image file: {file_path}")
            
            # Create image output directory
            image_output_dir = os.path.join(output_dir, "images")
            rel_image_dir = os.path.dirname(rel_path)
            target_dir = os.path.join(image_output_dir, rel_image_dir)
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy the image file to the images directory
            output_image_path = os.path.join(image_output_dir, rel_path)
            shutil.copy2(file_path, output_image_path)
            print(f"  Copied image -> {output_image_path}")
            
            # Log to Phoenix if enabled
            if observer and trace_id:
                # Generate a mock embedding for the image
                start_time = time.time()
                observer.log_transformation_span(
                    trace_id=trace_id,
                    transformation_type="image_processing",
                    file_path=file_path,
                    num_elements=1,
                    duration_ms=(time.time() - start_time) * 1000
                )
                
                # Log image embedding generation
                start_time = time.time()
                observer.log_embedding_span(
                    trace_id=trace_id,
                    embedding_model="clip-ViT-B-32",
                    text_or_image=os.path.basename(file_path),
                    embedding_type="image",
                    embedding_dim=512,
                    duration_ms=(time.time() - start_time) * 1000
                )
            
            return 0, 1, True
        
        # Check if it's a text file
        elif not file_path.endswith(('.txt', '.md', '.rst', '.csv')):
            print(f"Skipping non-text file: {file_path}")
            return 0, 0, False
        
        print(f"Processing file: {file_path}")
        
        # Start transformation timing
        transformation_start = time.time()
        
        # Read the file
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            file_content = file.read()
        
        # Process with or without chunking
        if use_chunking:
            # Split text into chunks
            chunks = contextual_chunker(file_content, chunk_size, chunk_overlap)
            
            # Process each chunk
            elements = []
            for i, chunk in enumerate(chunks):
                chunk_elements = simple_text_processor(chunk)
                # Add chunk metadata
                for element in chunk_elements:
                    element["metadata"] = {
                        "source_type": "text_chunk",
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    }
                elements.extend(chunk_elements)
        else:
            # Process without chunking
            elements = simple_text_processor(file_content)
        
        # Log transformation to Phoenix
        transformation_duration = (time.time() - transformation_start) * 1000
        if observer and trace_id:
            observer.log_transformation_span(
                trace_id=trace_id,
                transformation_type="text_chunking" if use_chunking else "text_processing",
                file_path=file_path,
                num_elements=len(elements),
                chunk_size=chunk_size if use_chunking else None,
                chunk_overlap=chunk_overlap if use_chunking else None,
                duration_ms=transformation_duration
            )
        
        # Generate embeddings
        embedding_start = time.time()
        for element in elements:
            # In a real implementation, this would use a proper embedding model
            element["embedding"] = get_mock_embedding(element["text"])
        embedding_duration = (time.time() - embedding_start) * 1000
        
        # Log embedding generation to Phoenix
        if observer and trace_id:
            # Log a sample of elements for brevity
            for element in elements[:min(3, len(elements))]:
                observer.log_embedding_span(
                    trace_id=trace_id,
                    embedding_model="all-MiniLM-L6-v2",
                    text_or_image=element["text"],
                    embedding_type="text",
                    embedding_dim=len(element["embedding"]),
                    duration_ms=embedding_duration / len(elements)  # Average per element
                )
        
        # Output JSON path
        output_file = os.path.join(output_dir, f"{rel_path}.json")
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the processed output
        with open(output_file, "w") as f:
            # Remove actual embeddings before saving (to save space)
            output_elements = []
            for element in elements:
                output_element = element.copy()
                if "embedding" in output_element:
                    # Just note the embedding dimension instead of saving the full vector
                    output_element["embedding_dim"] = len(output_element["embedding"])
                    del output_element["embedding"]
                output_elements.append(output_element)
            
            json.dump(output_elements, f, indent=2)
        
        print(f"  Processed {len(elements)} elements -> {output_file}")
        return len(elements), len(elements), False
    except Exception as e:
        print(f"  Error processing {file_path}: {str(e)}")
        if observer and trace_id:
            observer.log_transformation_span(
                trace_id=trace_id,
                transformation_type="file_processing",
                file_path=file_path,
                num_elements=0,
                status="error",
                error=str(e)
            )
        return 0, 0, False

def create_sample_files(input_dir):
    """Create sample files for testing if the directory is empty."""
    # Create directory if it doesn't exist
    os.makedirs(input_dir, exist_ok=True)
    
    # Check if directory is empty
    if next(os.scandir(input_dir), None) is not None:
        return
    
    print("Creating sample files for testing...")
    
    # Create some sample files
    samples = [
        {"path": "sample1.txt", "content": "Sample Document 1\n\nThis is a sample document with multiple paragraphs.\n\nIt demonstrates how the processor handles text files."},
        {"path": "sample2.txt", "content": "Sample Document 2\n\nThis document has a list:\n- Item 1\n- Item 2\n- Item 3\n\nAnd some more text."},
        {"path": "subdir/sample3.txt", "content": "Nested Sample Document\n\nThis file is in a subdirectory to demonstrate recursive processing."},
    ]
    
    for sample in samples:
        file_path = os.path.join(input_dir, sample["path"])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(sample["content"])
    
    print(f"Created {len(samples)} sample files.")

def safe_clear_directory(directory):
    """Safely clear a directory without deleting the directory itself."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        try:
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Error removing {item_path}: {e}")

def process_directory(
    input_dir: str, 
    output_dir: str, 
    recursive: bool = True,
    use_chunking: bool = True,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    observer: Optional[PhoenixObserver] = None
) -> Tuple[int, int, int, int]:
    """
    Process all files in a directory and optionally its subdirectories.
    
    Args:
        input_dir: Directory containing files to process
        output_dir: Directory to save processed output to
        recursive: Whether to process subdirectories
        use_chunking: Whether to use contextual chunking
        chunk_size: Size of chunks for chunking
        chunk_overlap: Overlap between chunks
        observer: Optional Phoenix observer for monitoring
        
    Returns:
        Tuple of (text_file_count, element_count, image_file_count, embedding_count)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create image directory
    image_output_dir = os.path.join(output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Create sample files if directory is empty
    create_sample_files(input_dir)
    
    # Initialize counts
    text_file_count = 0
    image_file_count = 0
    element_count = 0
    embedding_count = 0
    
    # Create a trace if Phoenix is enabled
    trace_id = None
    if observer:
        trace_id = observer.start_trace(
            metadata={
                "input_dir": input_dir,
                "output_dir": output_dir,
                "recursive": recursive,
                "use_chunking": use_chunking,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
        )
    
    # Log extraction to Phoenix
    extraction_start = time.time()
    
    # Safely clear existing output directory
    safe_clear_directory(output_dir)
    # Recreate image directory after clearing
    os.makedirs(image_output_dir, exist_ok=True)
    
    try:
        # Get list of files to process
        all_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
            if not recursive:
                break
        
        # Log extraction to Phoenix
        extraction_duration = (time.time() - extraction_start) * 1000
        if observer and trace_id:
            observer.log_extraction_span(
                trace_id=trace_id,
                source_type="local_dir",
                source_details={"path": input_dir, "recursive": recursive},
                num_files=len(all_files),
                duration_ms=extraction_duration
            )
        
        # Process each file
        for file_path in all_files:
            elements, embeddings, is_image = process_file(
                file_path=file_path,
                output_dir=output_dir,
                input_base_dir=input_dir,
                observer=observer,
                trace_id=trace_id,
                use_chunking=use_chunking,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            if is_image:
                image_file_count += 1
                embedding_count += embeddings
            elif elements > 0:
                text_file_count += 1
                element_count += elements
                embedding_count += embeddings
        
        # Log loading to Phoenix
        if observer and trace_id:
            observer.log_loading_span(
                trace_id=trace_id,
                destination_type="local_dir",
                destination_details={"path": output_dir},
                num_elements=element_count + image_file_count
            )
            
            # End trace successfully
            observer.end_trace(trace_id, status="success")
    
    except Exception as e:
        print(f"Error processing directory: {str(e)}")
        
        # Log error to Phoenix
        if observer and trace_id:
            observer.end_trace(trace_id, status="error", error=str(e))
    
    return text_file_count, element_count, image_file_count, embedding_count

def main():
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up input and output paths relative to script directory
    input_dir = os.path.join(script_dir, "data")
    output_dir = os.path.join(script_dir, "processed_data")
    
    # Configure chunking options
    use_chunking = True  # Set to False to disable contextual chunking
    chunk_size = 1000
    chunk_overlap = 200
    
    # Initialize Phoenix observer
    try:
        phoenix_enabled = True
        observer = PhoenixObserver(
            enabled=phoenix_enabled,
            app_name="ai-agent-etl-pipeline",
            env="development"
        )
        print("Phoenix observability enabled")
    except Exception as e:
        print(f"Could not initialize Phoenix observer: {str(e)}")
        observer = None
    
    print(f"Starting ETL process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Contextual chunking: {'Enabled' if use_chunking else 'Disabled'}")
    if use_chunking:
        print(f"  Chunk size: {chunk_size} characters")
        print(f"  Chunk overlap: {chunk_overlap} characters")
    
    # Process all files in the directory
    start_time = time.time()
    text_file_count, element_count, image_file_count, embedding_count = process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=True,
        use_chunking=use_chunking,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        observer=observer
    )
    processing_time = time.time() - start_time
    
    # Print summary
    print("\nETL Process Complete")
    print(f"Processed {text_file_count} text files")
    print(f"Copied {image_file_count} image files")
    print(f"Extracted {element_count} elements")
    print(f"Generated {embedding_count} embeddings")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print(f"Images saved to: {os.path.join(output_dir, 'images')}")
    
    if observer:
        print(f"Telemetry data sent to Phoenix")
        print("View results in Phoenix UI")

if __name__ == "__main__":
    main()