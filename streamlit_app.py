#!/usr/bin/env python3
"""
Ultra-minimal Streamlit app for ETL visualization - designed for Streamlit Cloud compatibility.
This version has minimal dependencies and contains all necessary code within one file.
"""
import os
import time
import uuid
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="AI ETL Pipeline Visualizer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'processed_elements' not in st.session_state:
    st.session_state.processed_elements = []
if 'embedding_model_loaded' not in st.session_state:
    st.session_state.embedding_model_loaded = False

# Simplified functions that won't require external dependencies
def simple_text_processor(text, use_chunking=True, chunk_size=1000, chunk_overlap=200):
    """Process text into elements, optionally with chunking."""
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
    
    if use_chunking:
        # Simplified chunking
        full_text = text
        chunks = []
        start = 0
        
        while start < len(full_text):
            # Find end of chunk
            end = min(start + chunk_size, len(full_text))
            
            # Try to break at paragraph
            last_break = full_text.rfind('\n\n', start, end)
            if last_break != -1 and last_break > start + chunk_size // 2:
                end = last_break
            
            # Extract chunk
            chunk = full_text[start:end].strip()
            if chunk:
                elements.append({
                    "type": "Chunk",
                    "text": chunk,
                    "element_id": f"chunk-{uuid.uuid4()}",
                    "metadata": {
                        "source_type": "text_chunk", 
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                    }
                })
            
            # Move start with overlap
            start = end - chunk_overlap if end - chunk_overlap > start else end
    else:
        # Simple paragraph processing
        paragraphs = text.split('\n\n')
        for i, para in enumerate(paragraphs):
            if para.strip():
                elements.append({
                    "type": "Text",
                    "text": para.strip(),
                    "element_id": f"element-{uuid.uuid4()}",
                    "metadata": {"source_type": "paragraph", "index": i}
                })
    
    return elements

def mock_generate_embedding(text_or_image, dimension=384):
    """Generate a mock embedding for demo purposes."""
    # Create a deterministic but somewhat unique embedding based on input content
    if isinstance(text_or_image, str):
        # Use the text length as seed
        seed = len(text_or_image)
    else:
        # For images (PIL Image), use size as seed
        seed = sum(text_or_image.size)
    
    # Generate a pseudo-random but deterministic embedding
    np.random.seed(seed)
    return np.random.normal(0, 1, dimension).tolist()

def try_load_sentence_transformer():
    """Try to load the sentence transformer model."""
    try:
        # Wrap this in another try block to catch any errors during import or initialization
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            st.session_state.embedding_model = model
            st.session_state.embedding_model_loaded = True
            return True
        except Exception as e:
            st.warning(f"Error loading sentence-transformers: {str(e)}")
            return False
    except ImportError:
        st.warning("Could not import sentence-transformers - will use mock embeddings instead")
        return False

# Sidebar
with st.sidebar:
    st.title("AI ETL Pipeline")
    
    # Chunking options
    st.header("Chunking Options")
    use_chunking = st.checkbox("Use Contextual Chunking", value=True)
    chunk_size = st.slider("Chunk Size (characters)", min_value=100, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=500, value=200, step=50)
    
    # Try to load the real embedding model
    if st.button("Load Embedding Model (Optional)"):
        with st.spinner("Attempting to load model..."):
            success = try_load_sentence_transformer()
            if success:
                st.success("Model loaded successfully!")
            else:
                st.info("Using mock embeddings for demonstration")
    
    # About section
    st.header("About")
    st.markdown("""
    This app demonstrates the AI ETL Pipeline with vector embeddings. It provides:
    
    - Text processing with contextual chunking
    - Embedding generation (mock or real)
    - Basic visualization
    
    This version is designed to work on Streamlit Cloud with minimal dependencies.
    """)

# Main content
st.title("AI ETL Pipeline Visualizer")
st.markdown("""
This simplified app demonstrates the key concepts of the ETL pipeline with vector embeddings.
Enter text below to see how it gets processed and embedded.
""")

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
                
                # Generate embeddings
                for element in elements:
                    if st.session_state.embedding_model_loaded:
                        element['embedding'] = st.session_state.embedding_model.encode(element['text']).tolist()
                    else:
                        element['embedding'] = mock_generate_embedding(element['text'])
                
                # Store in session state
                st.session_state.processed_elements = elements
                
                st.success(f"Processed text into {len(elements)} elements")
    
    with col2:
        st.subheader("Processed Elements")
        
        if 'processed_elements' in st.session_state and st.session_state.processed_elements:
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
            
            # Visualize an embedding
            if elements:
                st.subheader("Embedding Visualization")
                
                # Select which element to visualize
                element_idx = st.selectbox(
                    "Select element to visualize", 
                    range(len(elements)), 
                    format_func=lambda i: f"{elements[i]['type']}: {elements[i]['text'][:30]}..."
                )
                
                selected_element = elements[element_idx]
                
                # Plot embedding distribution
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.hist(selected_element['embedding'], bins=30)
                ax.set_title(f"Embedding Distribution for {selected_element['type']}")
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
                
                # Show first few values
                st.subheader("First 10 Embedding Dimensions")
                st.write(selected_element['embedding'][:10])

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
                    # Generate mock embedding
                    try:
                        # Try to use real image embedding if CLIP is available
                        try:
                            from sentence_transformers import SentenceTransformer
                            model = SentenceTransformer("clip-ViT-B-32")
                            embedding = model.encode(image).tolist()
                            is_real = True
                        except Exception as e:
                            st.warning(f"Error loading CLIP model: {str(e)}")
                            embedding = mock_generate_embedding(image, dimension=512)
                            is_real = False
                    except ImportError:
                        # Fall back to mock embedding
                        embedding = mock_generate_embedding(image, dimension=512)
                        is_real = False
                    
                    # Store in session state
                    st.session_state.image_embedding = embedding
                    st.session_state.is_real_image_embedding = is_real
                    
                    st.success(f"Generated {'real' if is_real else 'mock'} embedding with {len(embedding)} dimensions")
    
    with col2:
        st.subheader("Image Embedding")
        
        if 'image_embedding' in st.session_state:
            embedding = st.session_state.image_embedding
            is_real = st.session_state.get('is_real_image_embedding', False)
            
            st.info(f"Using {'real CLIP' if is_real else 'mock'} embeddings")
            
            # Show stats
            st.metric("Embedding Dimensions", len(embedding))
            st.metric("Min Value", round(min(embedding), 3))
            st.metric("Max Value", round(max(embedding), 3))
            
            # Plot embedding distribution
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(embedding, bins=50)
            ax.set_title("Embedding Value Distribution")
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            
            # Show first few values
            st.subheader("First 10 Dimensions")
            st.write(embedding[:10])

# Footer
st.markdown("---")
st.markdown("""
### About AI Agent ETL Pipeline

This demo showcases:
- Contextual chunking for dividing text into semantic units
- Vector embeddings for text and images
- Visualization of embedding distributions

For the full experience with Weaviate integration and advanced search capabilities,
please run the complete application locally with Docker.
""")

st.caption("Created with Streamlit and AI Agent ETL Pipeline")