#!/usr/bin/env python3
"""
Script to process text and image files, generate embeddings, and store in Weaviate.
This demonstrates a complete local ETL process with vector database integration.
"""
import os
import json
import re
import shutil
import uuid
import time
import io
from pathlib import Path
from datetime import datetime
import requests
from typing import List, Dict, Any, Optional, Union

# For local embedding generation
from sentence_transformers import SentenceTransformer
from PIL import Image

# For Weaviate integration
import weaviate
from weaviate.util import generate_uuid5

# Global embedding models (loaded once)
text_model = None
image_model = None

def load_text_embedding_model(model_name="all-MiniLM-L6-v2"):
    """Load the text embedding model once and reuse it."""
    global text_model
    if text_model is None:
        print(f"Loading text embedding model: {model_name}")
        text_model = SentenceTransformer(model_name)
    return text_model

def load_image_embedding_model(model_name="clip-ViT-B-32"):
    """Load the image embedding model once and reuse it."""
    global image_model
    if image_model is None:
        print(f"Loading image embedding model: {model_name}")
        image_model = SentenceTransformer(model_name)
    return image_model

def get_text_embedding(text: str) -> List[float]:
    """Generate an embedding for the given text."""
    model = load_text_embedding_model()
    return model.encode(text).tolist()

def get_image_embedding(image_path: str) -> List[float]:
    """Generate an embedding for the given image."""
    try:
        model = load_image_embedding_model()
        img = Image.open(image_path)
        return model.encode(img).tolist()
    except Exception as e:
        print(f"Error generating image embedding: {str(e)}")
        # Return empty embedding if there's an error
        return []

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

def simple_text_processor(text, use_chunking=True, chunk_size=1000, chunk_overlap=200):
    """
    A text processor that breaks text into elements and can perform contextual chunking.
    This is a simplified version of what the Unstructured API might do.
    """
    elements = []
    
    # Extract title (first line)
    lines = text.split('\n')
    if lines and lines[0].strip():
        title = lines[0].strip()
        elements.append({
            "type": "Title",
            "text": title,
            "element_id": f"element-{uuid.uuid4()}",
            "metadata": {"source_type": "title"}
        })
    else:
        title = ""
    
    # Join the rest of the text
    content = "\n".join(lines[1:]) if len(lines) > 1 else ""
    
    # Process paragraphs (separated by blank lines)
    paragraphs = re.split(r'\n\s*\n', content)
    
    if use_chunking:
        # Join paragraphs with their original separators for chunking
        full_content = title + "\n\n" + content if title else content
        
        # Generate overlapping chunks with context
        chunks = contextual_chunker(full_content, chunk_size, chunk_overlap)
        
        # Process each chunk as a separate element
        for i, chunk in enumerate(chunks):
            elements.append({
                "type": "Chunk",
                "text": chunk,
                "element_id": f"chunk-{uuid.uuid4()}",
                "metadata": {
                    "source_type": "text_chunk", 
                    "index": i,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "total_chunks": len(chunks)
                }
            })
    else:
        # Traditional paragraph processing
        for i, para in enumerate(paragraphs):
            if para.strip():
                elements.append({
                    "type": "Text",
                    "text": para.strip(),
                    "element_id": f"element-{uuid.uuid4()}",
                    "metadata": {"source_type": "paragraph", "index": i}
                })
    
    return elements

def connect_to_weaviate(url="http://localhost:8080"):
    """Connect to a Weaviate instance."""
    try:
        client = weaviate.Client(url=url)
        return client
    except Exception as e:
        print(f"Error connecting to Weaviate: {str(e)}")
        return None

def setup_weaviate_schema(client):
    """Set up the schema in Weaviate for our document elements and images."""
    if client is None:
        return False
    
    # Check if our schema already exists
    try:
        schema = client.schema.get()
        class_names = [obj["class"] for obj in schema["classes"]] if "classes" in schema else []
        
        # If our classes exist, we're good
        if "DocumentElement" in class_names and "ImageElement" in class_names:
            print("DocumentElement and ImageElement schemas already exist")
            return True
    except:
        # If we get an error, we'll create the schema
        pass
    
    # Define the schema for our document elements and images
    schema = {
        "classes": [
            {
                "class": "DocumentElement",
                "description": "A document element from processed text files",
                "vectorizer": "none",  # We'll provide our own vectors
                "properties": [
                    {
                        "name": "text",
                        "dataType": ["text"],
                        "description": "The text content of the element"
                    },
                    {
                        "name": "type",
                        "dataType": ["string"],
                        "description": "The type of element (Title, Text, Chunk, etc.)"
                    },
                    {
                        "name": "source_file",
                        "dataType": ["string"],
                        "description": "The source file path"
                    },
                    {
                        "name": "element_id",
                        "dataType": ["string"],
                        "description": "Unique identifier for the element"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["object"],
                        "description": "Additional metadata about the element, including chunking information"
                    },
                    {
                        "name": "is_chunk",
                        "dataType": ["boolean"],
                        "description": "Whether this element is a chunk of a larger text"
                    },
                    {
                        "name": "chunk_index",
                        "dataType": ["int"],
                        "description": "Index of this chunk in the sequence"
                    },
                    {
                        "name": "total_chunks",
                        "dataType": ["int"],
                        "description": "Total number of chunks for this document"
                    }
                ]
            },
            {
                "class": "ImageElement",
                "description": "An image with embeddings",
                "vectorizer": "none",  # We'll provide our own vectors
                "properties": [
                    {
                        "name": "file_path",
                        "dataType": ["string"],
                        "description": "Path to the image file"
                    },
                    {
                        "name": "file_name",
                        "dataType": ["string"],
                        "description": "Name of the image file"
                    },
                    {
                        "name": "source_file",
                        "dataType": ["string"],
                        "description": "The source file path"
                    },
                    {
                        "name": "image_id",
                        "dataType": ["string"],
                        "description": "Unique identifier for the image"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["object"],
                        "description": "Additional metadata about the image"
                    }
                ]
            }
        ]
    }
    
    try:
        # Create the schema
        client.schema.create(schema)
        print("Created DocumentElement and ImageElement schemas in Weaviate")
        return True
    except Exception as e:
        print(f"Error creating schema: {str(e)}")
        return False

def store_element_in_weaviate(client, element, file_path):
    """Store a document element in Weaviate with its embedding."""
    if client is None:
        return False
    
    try:
        # Generate embedding for the element text
        embedding = get_text_embedding(element["text"])
        
        # Create a unique ID based on file path and element ID
        element_uuid = generate_uuid5(element["element_id"])
        
        # Get metadata
        metadata = element.get("metadata", {})
        
        # Check if this is a chunked element
        is_chunk = element["type"] == "Chunk"
        chunk_index = metadata.get("index", 0) if is_chunk else 0
        total_chunks = metadata.get("total_chunks", 1) if is_chunk else 1
        
        # Store the element with its embedding
        client.data_object.create(
            data_object={
                "text": element["text"],
                "type": element["type"],
                "source_file": file_path,
                "element_id": element["element_id"],
                "metadata": metadata,
                "is_chunk": is_chunk,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks
            },
            class_name="DocumentElement",
            uuid=element_uuid,
            vector=embedding
        )
        return True
    except Exception as e:
        print(f"Error storing element in Weaviate: {str(e)}")
        return False

def store_image_in_weaviate(client, image_path, rel_path):
    """Store an image in Weaviate with its embedding."""
    if client is None:
        return False
    
    try:
        # Generate embedding for the image
        embedding = get_image_embedding(image_path)
        
        # Skip if embedding failed
        if not embedding:
            print(f"  Skipping image (embedding failed): {image_path}")
            return False
        
        # Create unique image ID
        image_id = f"image-{uuid.uuid4()}"
        image_uuid = generate_uuid5(image_id)
        
        # Get file name from path
        file_name = os.path.basename(image_path)
        
        # Store the image with its embedding
        client.data_object.create(
            data_object={
                "file_path": image_path,
                "file_name": file_name,
                "source_file": rel_path,
                "image_id": image_id,
                "metadata": {
                    "type": "image",
                    "extension": os.path.splitext(file_name)[1].lower()
                }
            },
            class_name="ImageElement",
            uuid=image_uuid,
            vector=embedding
        )
        return True
    except Exception as e:
        print(f"Error storing image in Weaviate: {str(e)}")
        return False

def process_file(file_path, output_dir, input_base_dir, weaviate_client=None):
    """Process a single file, save output, and store in Weaviate if client provided."""
    try:
        # Create relative output path to maintain directory structure
        rel_path = os.path.relpath(file_path, input_base_dir)
        
        # Check if it's an image file
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp')):
            print(f"Processing image file: {file_path}")
            
            # Create image output directory
            image_output_dir = os.path.join(output_dir, "images")
            rel_image_dir = os.path.dirname(rel_path)
            target_dir = os.path.join(image_output_dir, rel_image_dir)
            os.makedirs(target_dir, exist_ok=True)
            
            # Copy the image file to the images directory
            output_image_path = os.path.join(image_output_dir, rel_path)
            shutil.copy2(file_path, output_image_path)
            print(f"  Copied image -> {output_image_path}")
            
            # Store image in Weaviate if client is provided
            weaviate_image_count = 0
            if weaviate_client is not None:
                if store_image_in_weaviate(weaviate_client, file_path, rel_path):
                    weaviate_image_count = 1
                    print(f"  Stored image with embedding in Weaviate")
            
            return 0, weaviate_image_count, 1
        
        # Check if it's a text file
        elif not file_path.endswith(('.txt', '.md', '.rst', '.csv')):
            print(f"Skipping non-text file: {file_path}")
            return 0, 0, 0
        
        print(f"Processing file: {file_path}")
        
        # Read the file
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            file_content = file.read()
        
        # Process the file
        elements = simple_text_processor(file_content)
        
        # Output JSON path
        output_file = os.path.join(output_dir, f"{rel_path}.json")
        
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save the processed output
        with open(output_file, "w") as f:
            json.dump(elements, f, indent=2)
        
        # Store elements in Weaviate if client is provided
        weaviate_count = 0
        if weaviate_client is not None:
            for element in elements:
                if store_element_in_weaviate(weaviate_client, element, rel_path):
                    weaviate_count += 1
        
        print(f"  Processed {len(elements)} elements -> {output_file}")
        if weaviate_count > 0:
            print(f"  Stored {weaviate_count} text elements in Weaviate")
        
        return len(elements), weaviate_count, 0
    except Exception as e:
        print(f"  Error processing {file_path}: {str(e)}")
        return 0, 0, 0

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

def check_weaviate_available(url="http://localhost:8080"):
    """Check if Weaviate is available at the given URL."""
    try:
        response = requests.get(f"{url}/.well-known/ready")
        return response.status_code == 200
    except:
        return False

def process_directory(input_dir, output_dir, recursive=True, use_weaviate=True):
    """Process all text files in a directory and optionally its subdirectories."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create image directory
    image_output_dir = os.path.join(output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Create sample files if directory is empty
    create_sample_files(input_dir)
    
    # Connect to Weaviate if requested
    weaviate_client = None
    weaviate_text_count = 0
    weaviate_image_count = 0
    
    if use_weaviate:
        # Check if Weaviate is available
        if check_weaviate_available():
            weaviate_client = connect_to_weaviate()
            if weaviate_client and setup_weaviate_schema(weaviate_client):
                print("Connected to Weaviate successfully")
            else:
                print("Failed to set up Weaviate schema, proceeding without vector storage")
                weaviate_client = None
        else:
            print("Warning: Weaviate not available, proceeding without vector storage")
    
    # Process files
    text_file_count = 0
    image_file_count = 0
    element_count = 0
    
    # Safely clear existing output directory
    safe_clear_directory(output_dir)
    # Recreate image directory after clearing
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Walk through directory and process files
    for root, dirs, files in os.walk(input_dir):
        # Process files in current directory
        for file in files:
            file_path = os.path.join(root, file)
            
            # Process all files (text and images)
            elements, weaviate_elements, is_image = process_file(file_path, output_dir, input_dir, weaviate_client)
            
            if is_image:
                image_file_count += 1
                weaviate_image_count += weaviate_elements
            elif elements > 0:
                text_file_count += 1
                element_count += elements
                weaviate_text_count += weaviate_elements
        
        # Stop if not recursive
        if not recursive:
            break
    
    return text_file_count, element_count, image_file_count, weaviate_text_count, weaviate_image_count

def main():
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up input and output paths relative to script directory
    input_dir = os.path.join(script_dir, "data")
    output_dir = os.path.join(script_dir, "processed_data")
    
    # Configuration options
    use_weaviate = True  # Set to False to disable Weaviate integration
    use_chunking = True  # Set to True to enable contextual chunking
    chunk_size = 1000    # Maximum size of each chunk in characters
    chunk_overlap = 200  # Overlap between consecutive chunks
    
    # Update the text_processor function with chunking parameters
    global simple_text_processor
    original_text_processor = simple_text_processor
    simple_text_processor = lambda text: original_text_processor(
        text, use_chunking=use_chunking, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    
    print(f"Starting ETL process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Weaviate integration: {'Enabled' if use_weaviate else 'Disabled'}")
    print(f"Contextual chunking: {'Enabled' if use_chunking else 'Disabled'}")
    if use_chunking:
        print(f"  Chunk size: {chunk_size} characters")
        print(f"  Chunk overlap: {chunk_overlap} characters")
    
    # Process all files in the directory
    start_time = time.time()
    text_file_count, element_count, image_file_count, weaviate_text_count, weaviate_image_count = process_directory(
        input_dir, output_dir, recursive=True, use_weaviate=use_weaviate
    )
    processing_time = time.time() - start_time
    
    # Print summary
    print("\nETL Process Complete")
    print(f"Processed {text_file_count} text files")
    print(f"Copied {image_file_count} image files")
    print(f"Extracted {element_count} elements")
    if use_weaviate:
        print(f"Stored {weaviate_text_count} text elements in Weaviate")
        print(f"Stored {weaviate_image_count} images in Weaviate")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print(f"Images saved to: {os.path.join(output_dir, 'images')}")
    
if __name__ == "__main__":
    main()