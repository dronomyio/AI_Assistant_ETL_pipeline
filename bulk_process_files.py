#!/usr/bin/env python3
"""
Script to process all text files in a directory and subdirectories.
This demonstrates a complete local ETL process that recursively processes files.
"""
import os
import json
import re
import shutil
from pathlib import Path
from datetime import datetime

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
            "element_id": "element-0"
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

def process_file(file_path, output_dir, input_base_dir):
    """Process a single file and save the output to the specified directory."""
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
            return 0
        
        # Check if it's a text file
        elif not file_path.endswith(('.txt', '.md', '.rst', '.csv')):
            print(f"Skipping non-text file: {file_path}")
            return 0
        
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
        
        print(f"  Processed {len(elements)} elements -> {output_file}")
        return len(elements)
    except Exception as e:
        print(f"  Error processing {file_path}: {str(e)}")
        return 0

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

def process_directory(input_dir, output_dir, recursive=True):
    """Process all text files in a directory and optionally its subdirectories."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create image directory
    image_output_dir = os.path.join(output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Create sample files if directory is empty
    create_sample_files(input_dir)
    
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
                elements = process_file(file_path, output_dir, input_dir)
                if elements:
                    text_file_count += 1
                    element_count += elements
        
        # Stop if not recursive
        if not recursive:
            break
    
    return text_file_count, element_count, image_file_count

def main():
    # Get current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up input and output paths relative to script directory
    input_dir = os.path.join(script_dir, "data")
    output_dir = os.path.join(script_dir, "processed_data")
    
    print(f"Starting ETL process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Process all files in the directory
    text_file_count, element_count, image_file_count = process_directory(input_dir, output_dir, recursive=True)
    
    # Print summary
    print("\nETL Process Complete")
    print(f"Processed {text_file_count} text files")
    print(f"Copied {image_file_count} image files")
    print(f"Extracted {element_count} elements")
    print(f"Results saved to: {output_dir}")
    print(f"Images saved to: {os.path.join(output_dir, 'images')}")
    
if __name__ == "__main__":
    main()