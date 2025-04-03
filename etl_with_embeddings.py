#!/usr/bin/env python3
"""
Script to process all text files, generate embeddings, and store in Weaviate.
This demonstrates a complete local ETL process with vector database integration.
"""
import os
import json
import re
import shutil
import uuid
import time
from pathlib import Path
from datetime import datetime
import requests
from typing import List, Dict, Any, Optional, Union

# For local embedding generation
from sentence_transformers import SentenceTransformer

# For Weaviate integration
import weaviate
from weaviate.util import generate_uuid5

# Global embedding model (loaded once)
model = None

def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    """Load the embedding model once and reuse it."""
    global model
    if model is None:
        print(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)
    return model

def get_embedding(text: str) -> List[float]:
    """Generate an embedding for the given text."""
    model = load_embedding_model()
    return model.encode(text).tolist()

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
            "element_id": f"element-{uuid.uuid4()}",
            "metadata": {"source_type": "title"}
        })
    
    # Process paragraphs (separated by blank lines)
    paragraphs = re.split(r'\n\s*\n', text)
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
    """Set up the schema in Weaviate for our document elements."""
    if client is None:
        return False
    
    # Check if our schema already exists
    try:
        schema = client.schema.get()
        class_names = [obj["class"] for obj in schema["classes"]] if "classes" in schema else []
        
        # If our class exists, we're good
        if "DocumentElement" in class_names:
            print("DocumentElement schema already exists")
            return True
    except:
        # If we get an error, we'll create the schema
        pass
    
    # Define the schema for our document elements
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
                        "description": "The type of element (Title, Text, etc.)"
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
                        "description": "Additional metadata about the element"
                    }
                ]
            }
        ]
    }
    
    try:
        # Create the schema
        client.schema.create(schema)
        print("Created DocumentElement schema in Weaviate")
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
        embedding = get_embedding(element["text"])
        
        # Create a unique ID based on file path and element ID
        element_uuid = generate_uuid5(element["element_id"])
        
        # Store the element with its embedding
        client.data_object.create(
            data_object={
                "text": element["text"],
                "type": element["type"],
                "source_file": file_path,
                "element_id": element["element_id"],
                "metadata": element.get("metadata", {})
            },
            class_name="DocumentElement",
            uuid=element_uuid,
            vector=embedding
        )
        return True
    except Exception as e:
        print(f"Error storing element in Weaviate: {str(e)}")
        return False

def process_file(file_path, output_dir, input_base_dir, weaviate_client=None):
    """Process a single file, save output, and store in Weaviate if client provided."""
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
            return 0, 0
        
        # Check if it's a text file
        elif not file_path.endswith(('.txt', '.md', '.rst', '.csv')):
            print(f"Skipping non-text file: {file_path}")
            return 0, 0
        
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
            print(f"  Stored {weaviate_count} elements in Weaviate")
        
        return len(elements), weaviate_count
    except Exception as e:
        print(f"  Error processing {file_path}: {str(e)}")
        return 0, 0

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
    weaviate_element_count = 0
    
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
            
            # Check if it's an image
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp')):
                process_file(file_path, output_dir, input_dir)
                image_file_count += 1
            else:
                # Process text and other files
                elements, weaviate_elements = process_file(file_path, output_dir, input_dir, weaviate_client)
                if elements:
                    text_file_count += 1
                    element_count += elements
                    weaviate_element_count += weaviate_elements
        
        # Stop if not recursive
        if not recursive:
            break
    
    return text_file_count, element_count, image_file_count, weaviate_element_count

def main():
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up input and output paths relative to script directory
    input_dir = os.path.join(script_dir, "data")
    output_dir = os.path.join(script_dir, "processed_data")
    
    # Check if Weaviate is enabled
    use_weaviate = True  # Set to False to disable Weaviate integration
    
    print(f"Starting ETL process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Weaviate integration: {'Enabled' if use_weaviate else 'Disabled'}")
    
    # Process all files in the directory
    start_time = time.time()
    text_file_count, element_count, image_file_count, weaviate_count = process_directory(
        input_dir, output_dir, recursive=True, use_weaviate=use_weaviate
    )
    processing_time = time.time() - start_time
    
    # Print summary
    print("\nETL Process Complete")
    print(f"Processed {text_file_count} text files")
    print(f"Copied {image_file_count} image files")
    print(f"Extracted {element_count} elements")
    if use_weaviate:
        print(f"Stored {weaviate_count} elements in Weaviate")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Results saved to: {output_dir}")
    print(f"Images saved to: {os.path.join(output_dir, 'images')}")
    
if __name__ == "__main__":
    main()