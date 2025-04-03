#!/usr/bin/env python3
"""
Minimal Streamlit app to visualize the ETL pipeline with embeddings and vector search.
This version avoids using Plotly to prevent dependency issues.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import weaviate
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

# Import functions from our ETL script if available
try:
    from etl_with_embeddings import (
        get_text_embedding, 
        get_image_embedding, 
        simple_text_processor,
        contextual_chunker,
        check_weaviate_available, 
        connect_to_weaviate,
        setup_weaviate_schema
    )
except ImportError:
    # Define minimal versions if import fails
    def get_text_embedding(text):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(text).tolist()
        
    def get_image_embedding(image_path):
        return [0] * 512  # Return dummy embedding
        
    def simple_text_processor(text, use_chunking=True, chunk_size=1000, chunk_overlap=200):
        elements = []
        # Extract title (first line)
        lines = text.split('\n')
        if lines and lines[0].strip():
            elements.append({
                "type": "Title",
                "text": lines[0].strip(),
                "element_id": "element-0",
                "metadata": {"source_type": "title"}
            })
        
        # Process paragraphs (separated by blank lines)
        paragraphs = text.split('\n\n')
        for i, para in enumerate(paragraphs):
            if para.strip():
                elements.append({
                    "type": "Text",
                    "text": para.strip(),
                    "element_id": f"element-{i+1}",
                    "metadata": {"source_type": "paragraph", "index": i}
                })
        return elements
        
    def contextual_chunker(text, chunk_size=1000, chunk_overlap=200):
        return [text]
        
    def check_weaviate_available(url="http://localhost:8080"):
        return False
        
    def connect_to_weaviate(url="http://localhost:8080"):
        return None
        
    def setup_weaviate_schema(client):
        return False

# Set page config
st.set_page_config(
    page_title="AI ETL Pipeline Visualizer (Minimal)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'weaviate_client' not in st.session_state:
    st.session_state.weaviate_client = None
if 'text_model' not in st.session_state:
    st.session_state.text_model = None

# Sidebar
with st.sidebar:
    st.title("AI ETL Pipeline")
    
    # Weaviate connection
    st.header("Weaviate Connection")
    weaviate_url = st.text_input("Weaviate URL", value="http://localhost:8080")
    
    if st.button("Connect to Weaviate"):
        if check_weaviate_available(weaviate_url):
            st.session_state.weaviate_client = connect_to_weaviate(weaviate_url)
            if st.session_state.weaviate_client and setup_weaviate_schema(st.session_state.weaviate_client):
                st.success("Connected to Weaviate successfully!")
            else:
                st.error("Failed to set up Weaviate schema")
        else:
            st.error("Could not connect to Weaviate. Is it running?")
    
    # Chunking options
    st.header("Chunking Options")
    use_chunking = st.checkbox("Use Contextual Chunking", value=True)
    chunk_size = st.slider("Chunk Size (characters)", min_value=100, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=500, value=200, step=50)
    
    # Load embedding model
    if st.button("Load Text Embedding Model"):
        with st.spinner("Loading model..."):
            st.session_state.text_model = SentenceTransformer("all-MiniLM-L6-v2")
            st.success("Model loaded successfully!")
    
    # About section
    st.header("About")
    st.markdown("""
    This is a minimal version of the app that visualizes the AI ETL Pipeline with vector embeddings and search.
    """)

# Main content
st.title("AI ETL Pipeline Visualizer (Minimal Version)")

# Tabs
tab1, tab2 = st.tabs(["Process Text", "Process Images"])

# Process Text tab
with tab1:
    st.header("Process Text Documents")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Text")
        text_input = st.text_area("Enter text to process", height=300, 
                                 value="This is a sample document that will be processed.\n\nIt contains multiple paragraphs that demonstrate how the system works with different types of content.\n\nThe contextual chunking algorithm will split this text into overlapping segments while preserving natural boundaries like paragraph breaks.")
        
        process_button = st.button("Process Text")
        
        if process_button and text_input:
            with st.spinner("Processing text..."):
                # Process text with or without chunking
                elements = simple_text_processor(
                    text_input, 
                    use_chunking=use_chunking,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                # Generate embeddings if model is loaded
                if st.session_state.text_model is not None:
                    for element in elements:
                        element['embedding'] = st.session_state.text_model.encode(element['text']).tolist()
                else:
                    # Add dummy embeddings
                    for element in elements:
                        element['embedding'] = [0] * 384
                
                # Store in session state
                st.session_state.processed_elements = elements
                
                st.success(f"Processed text into {len(elements)} elements")
    
    with col2:
        st.subheader("Processed Elements")
        
        if 'processed_elements' in st.session_state:
            elements = st.session_state.processed_elements
            
            # Display as table
            element_data = []
            for i, element in enumerate(elements):
                element_type = element['type']
                text = element['text'][:100] + "..." if len(element['text']) > 100 else element['text']
                element_data.append({
                    "Index": i,
                    "Type": element_type,
                    "Text": text,
                    "Embedding Dim": len(element['embedding'])
                })
            
            st.dataframe(element_data)
            
            # Show raw JSON
            with st.expander("View Raw JSON"):
                # Remove embeddings to make the display cleaner
                display_elements = []
                for element in elements:
                    element_copy = element.copy()
                    if 'embedding' in element_copy:
                        element_copy['embedding'] = f"[{len(element_copy['embedding'])} dimensions]"
                    display_elements.append(element_copy)
                
                st.json(display_elements)

# Process Images tab
with tab2:
    st.header("Process Images")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            process_image_button = st.button("Generate Image Embedding")
            
            if process_image_button:
                with st.spinner("Processing image..."):
                    # Save temporary file
                    temp_path = f"temp_image_{int(time.time())}.jpg"
                    image.save(temp_path)
                    
                    try:
                        # Generate embedding
                        embedding = get_image_embedding(temp_path)
                        
                        # Store in session state
                        st.session_state.image_embedding = embedding
                        st.session_state.processed_image = {
                            "file_name": uploaded_file.name,
                            "embedding": embedding
                        }
                        
                        st.success(f"Generated embedding with {len(embedding)} dimensions")
                    except Exception as e:
                        st.error(f"Error generating embedding: {str(e)}")
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    with col2:
        st.subheader("Image Embedding")
        
        if 'image_embedding' in st.session_state:
            embedding = st.session_state.image_embedding
            
            # Show stats
            st.metric("Embedding Dimensions", len(embedding))
            
            # Plot embedding distribution if non-zero
            if any(embedding):
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(embedding, bins=50)
                ax.set_title("Embedding Value Distribution")
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
            
            # Show first few values
            st.subheader("First 10 Dimensions")
            st.write(embedding[:10])

# Footer with instructions
st.markdown("---")
st.info("This is a minimal version of the application. For the full experience with embedding visualization and search functionality, please ensure all dependencies are correctly installed.")