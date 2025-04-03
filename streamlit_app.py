#!/usr/bin/env python3
"""
Streamlit app to visualize the ETL pipeline with embeddings and vector search.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
import weaviate
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import io

# Import functions from our ETL script
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
    st.error("Could not import functions from etl_with_embeddings.py. Make sure the file exists in the same directory.")
    st.stop()

# Set page config
st.set_page_config(
    page_title="AI ETL Pipeline Visualizer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'weaviate_client' not in st.session_state:
    st.session_state.weaviate_client = None
if 'document_count' not in st.session_state:
    st.session_state.document_count = 0
if 'image_count' not in st.session_state:
    st.session_state.image_count = 0
if 'chunk_count' not in st.session_state:
    st.session_state.chunk_count = 0
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
                
                # Get counts
                try:
                    result = st.session_state.weaviate_client.query.aggregate("DocumentElement").with_fields("meta {count}").do()
                    if result and "data" in result and "Aggregate" in result["data"] and "DocumentElement" in result["data"]["Aggregate"]:
                        st.session_state.document_count = result["data"]["Aggregate"]["DocumentElement"][0]["meta"]["count"]
                    
                    result = st.session_state.weaviate_client.query.aggregate("ImageElement").with_fields("meta {count}").do()
                    if result and "data" in result and "Aggregate" in result["data"] and "ImageElement" in result["data"]["Aggregate"]:
                        st.session_state.image_count = result["data"]["Aggregate"]["ImageElement"][0]["meta"]["count"]
                        
                    # Get count of chunks
                    result = st.session_state.weaviate_client.query.get(
                        "DocumentElement",
                        ["element_id"]
                    ).with_where({
                        "path": ["is_chunk"],
                        "operator": "Equal",
                        "valueBoolean": True
                    }).with_limit(1).do()
                    
                    if result and "data" in result and "Get" in result["data"] and "DocumentElement" in result["data"]["Get"]:
                        result = st.session_state.weaviate_client.query.aggregate("DocumentElement").with_where({
                            "path": ["is_chunk"],
                            "operator": "Equal",
                            "valueBoolean": True
                        }).with_fields("meta {count}").do()
                        
                        if result and "data" in result and "Aggregate" in result["data"] and "DocumentElement" in result["data"]["Aggregate"]:
                            st.session_state.chunk_count = result["data"]["Aggregate"]["DocumentElement"][0]["meta"]["count"]
                        
                except Exception as e:
                    st.warning(f"Connected, but couldn't get counts: {str(e)}")
            else:
                st.error("Failed to set up Weaviate schema")
        else:
            st.error("Could not connect to Weaviate. Is it running?")
    
    if st.session_state.weaviate_client:
        st.info(f"Connected to Weaviate at {weaviate_url}")
        st.metric("Document Elements", st.session_state.document_count)
        st.metric("Image Elements", st.session_state.image_count)
        st.metric("Text Chunks", st.session_state.chunk_count)
    
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
    This app visualizes the AI ETL Pipeline with vector embeddings and search.
    
    Features:
    - Process text and generate embeddings
    - Process images and generate embeddings
    - Visualize embedding clusters
    - Search for similar content
    """)

# Main content
st.title("AI ETL Pipeline Visualizer")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Process Text", "Process Images", "Explore Embeddings", "Search"])

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
                    element['embedding'] = get_text_embedding(element['text'])
                
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
            
            # Show full embedding option
            with st.expander("View Full Embedding"):
                st.write(embedding)
            
            if st.session_state.weaviate_client:
                if st.button("Store in Weaviate"):
                    with st.spinner("Storing in Weaviate..."):
                        # Create a unique ID
                        import uuid
                        from weaviate.util import generate_uuid5
                        
                        image_id = f"image-{uuid.uuid4()}"
                        image_uuid = generate_uuid5(image_id)
                        
                        try:
                            # Store the image with its embedding
                            st.session_state.weaviate_client.data_object.create(
                                data_object={
                                    "file_name": uploaded_file.name,
                                    "source_file": "uploaded_via_ui",
                                    "image_id": image_id,
                                    "metadata": {
                                        "type": "image",
                                        "extension": os.path.splitext(uploaded_file.name)[1].lower(),
                                        "source": "streamlit_app"
                                    }
                                },
                                class_name="ImageElement",
                                uuid=image_uuid,
                                vector=embedding
                            )
                            st.success("Image stored in Weaviate!")
                            st.session_state.image_count += 1
                        except Exception as e:
                            st.error(f"Error storing in Weaviate: {str(e)}")

# Explore Embeddings tab
with tab3:
    st.header("Explore Embeddings")
    
    if not st.session_state.weaviate_client:
        st.warning("Please connect to Weaviate first to explore embeddings")
    else:
        st.subheader("Embedding Visualization")
        
        # Options
        visualization_type = st.radio("Visualization Type", ["PCA", "UMAP"])
        element_count = st.slider("Number of elements to visualize", min_value=50, max_value=500, value=100, step=50)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Element type selection
            element_types = st.multiselect(
                "Element Types", 
                ["Title", "Text", "Chunk", "MergedChunks", "Image"], 
                default=["Title", "Text", "Chunk"]
            )
            
            # Generate visualization button
            if st.button("Generate Visualization"):
                with st.spinner("Fetching data and generating visualization..."):
                    # Collect data for each type
                    all_data = []
                    all_embeddings = []
                    all_labels = []
                    
                    # Get text elements
                    if any(t in element_types for t in ["Title", "Text", "Chunk", "MergedChunks"]):
                        where_clause = {
                            "operator": "Or",
                            "operands": []
                        }
                        
                        for element_type in element_types:
                            if element_type != "Image":
                                where_clause["operands"].append({
                                    "path": ["type"],
                                    "operator": "Equal",
                                    "valueString": element_type
                                })
                        
                        if where_clause["operands"]:
                            result = st.session_state.weaviate_client.query.get(
                                "DocumentElement", 
                                ["text", "type", "source_file", "element_id", "_additional {vector}"]
                            ).with_where(where_clause).with_limit(element_count).do()
                            
                            if "data" in result and "Get" in result["data"] and "DocumentElement" in result["data"]["Get"]:
                                for item in result["data"]["Get"]["DocumentElement"]:
                                    if "_additional" in item and "vector" in item["_additional"]:
                                        all_data.append(item)
                                        all_embeddings.append(item["_additional"]["vector"])
                                        all_labels.append(item["type"])
                    
                    # Get image elements
                    if "Image" in element_types:
                        image_count = min(element_count // 2, 50)  # Limit number of images to avoid overloading
                        
                        result = st.session_state.weaviate_client.query.get(
                            "ImageElement", 
                            ["file_name", "source_file", "image_id", "_additional {vector}"]
                        ).with_limit(image_count).do()
                        
                        if "data" in result and "Get" in result["data"] and "ImageElement" in result["data"]["Get"]:
                            for item in result["data"]["Get"]["ImageElement"]:
                                if "_additional" in item and "vector" in item["_additional"]:
                                    item["type"] = "Image"  # Add type field
                                    all_data.append(item)
                                    all_embeddings.append(item["_additional"]["vector"])
                                    all_labels.append("Image")
                    
                    # Store in session state
                    st.session_state.viz_data = all_data
                    st.session_state.viz_embeddings = all_embeddings
                    st.session_state.viz_labels = all_labels
        
        with col2:
            if 'viz_embeddings' in st.session_state and st.session_state.viz_embeddings:
                embeddings = np.array(st.session_state.viz_embeddings)
                labels = st.session_state.viz_labels
                
                # Dimensionality reduction
                if visualization_type == "PCA":
                    pca = PCA(n_components=2)
                    reduced_embeds = pca.fit_transform(embeddings)
                    var_explained = pca.explained_variance_ratio_
                    title = f"PCA Visualization (Explained variance: {var_explained[0]:.2%}, {var_explained[1]:.2%})"
                else:  # UMAP
                    reducer = umap.UMAP(random_state=42)
                    reduced_embeds = reducer.fit_transform(embeddings)
                    title = "UMAP Visualization"
                
                # Create dataframe for plotting
                df = pd.DataFrame({
                    'x': reduced_embeds[:, 0],
                    'y': reduced_embeds[:, 1],
                    'type': labels
                })
                
                # Create plot
                fig = px.scatter(
                    df, x='x', y='y', color='type',
                    title=title,
                    labels={'x': 'Dimension 1', 'y': 'Dimension 2'},
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display stats
                st.subheader("Embedding Statistics")
                type_counts = df['type'].value_counts().reset_index()
                type_counts.columns = ['Element Type', 'Count']
                st.dataframe(type_counts)
            else:
                st.info("Click 'Generate Visualization' to see embeddings plotted here")

# Search tab
with tab4:
    st.header("Vector Search")
    
    if not st.session_state.weaviate_client:
        st.warning("Please connect to Weaviate first to perform searches")
    else:
        # Search options
        search_type = st.radio("Search Type", ["Text Search", "Image Search", "Combined Search"])
        
        if search_type == "Text Search":
            search_query = st.text_input("Enter your search query", "drone camera setup")
            group_chunks = st.checkbox("Group chunks from same document", value=True)
            result_limit = st.slider("Number of results", min_value=3, max_value=20, value=5)
            
            if st.button("Search"):
                with st.spinner("Searching..."):
                    # Generate embedding for the query
                    if st.session_state.text_model is None:
                        st.session_state.text_model = SentenceTransformer("all-MiniLM-L6-v2")
                    
                    query_embedding = st.session_state.text_model.encode(search_query).tolist()
                    
                    # Search Weaviate for similar text
                    try:
                        # Get more results if we're grouping chunks
                        actual_limit = result_limit * 3 if group_chunks else result_limit
                        
                        result = st.session_state.weaviate_client.query.get(
                            "DocumentElement", 
                            ["text", "type", "source_file", "element_id", "is_chunk", "chunk_index", 
                             "total_chunks", "metadata", "_additional {certainty}"]
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
                                    
                                    # Get average certainty
                                    certainties = [c.get("_additional", {}).get("certainty", 0) for c in chunks if "_additional" in c]
                                    avg_certainty = sum(certainties) / len(certainties) if certainties else 0
                                    
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
                                        },
                                        "_additional": {
                                            "certainty": avg_certainty
                                        }
                                    }
                                    
                                    final_results.append(merged_result)
                                
                                # Limit final results
                                results = final_results[:result_limit]
                            else:
                                results = results[:result_limit]
                            
                            # Display results
                            st.subheader(f"Found {len(results)} results")
                            
                            for i, item in enumerate(results):
                                with st.expander(f"Result {i+1}: {item['type']} - {item['source_file']}"):
                                    # Show relevance score
                                    if "_additional" in item and "certainty" in item["_additional"]:
                                        st.progress(item["_additional"]["certainty"])
                                        st.caption(f"Relevance: {item['_additional']['certainty']:.2%}")
                                    
                                    # Handle merged chunks differently
                                    if item['type'] == "MergedChunks" and "metadata" in item and "chunk_count" in item["metadata"]:
                                        chunk_count = item["metadata"]["chunk_count"]
                                        st.caption(f"[Merged from {chunk_count} chunks]")
                                    
                                    # Show text content
                                    st.markdown(item['text'])
                        else:
                            st.warning("No results found")
                    except Exception as e:
                        st.error(f"Error searching Weaviate: {str(e)}")
        
        elif search_type == "Image Search":
            # Provide two options - text description or upload image
            image_search_method = st.radio("Search Method", ["By Text Description", "By Similarity to Image"])
            
            if image_search_method == "By Text Description":
                image_query = st.text_input("Describe the image you're looking for", "drone camera setup")
                image_result_limit = st.slider("Number of results", min_value=3, max_value=20, value=5)
                
                if st.button("Search Images"):
                    with st.spinner("Searching for images..."):
                        # Generate embedding for the text query
                        if st.session_state.text_model is None:
                            st.session_state.text_model = SentenceTransformer("all-MiniLM-L6-v2")
                        
                        query_embedding = st.session_state.text_model.encode(image_query).tolist()
                        
                        # Search Weaviate for similar images
                        try:
                            result = st.session_state.weaviate_client.query.get(
                                "ImageElement", 
                                ["file_path", "file_name", "source_file", "image_id", "metadata", "_additional {certainty}"]
                            ).with_near_vector({
                                "vector": query_embedding
                            }).with_limit(image_result_limit).do()
                            
                            if "data" in result and "Get" in result["data"] and "ImageElement" in result["data"]["Get"]:
                                results = result["data"]["Get"]["ImageElement"]
                                
                                # Display results
                                st.subheader(f"Found {len(results)} images")
                                
                                cols = st.columns(3)
                                for i, item in enumerate(results):
                                    with cols[i % 3]:
                                        # Show relevance score
                                        if "_additional" in item and "certainty" in item["_additional"]:
                                            st.progress(item["_additional"]["certainty"])
                                            st.caption(f"Relevance: {item['_additional']['certainty']:.2%}")
                                        
                                        # Show image details
                                        st.caption(f"File: {item['file_name']}")
                                        st.caption(f"Source: {item['source_file']}")
                                        
                                        # Try to display the actual image if available
                                        try:
                                            if "file_path" in item and os.path.exists(item["file_path"]):
                                                img = Image.open(item["file_path"])
                                                st.image(img, caption=item['file_name'], use_column_width=True)
                                            else:
                                                st.info("Image file not accessible for preview")
                                        except Exception:
                                            st.info("Image preview not available")
                            else:
                                st.warning("No image results found")
                        except Exception as e:
                            st.error(f"Error searching Weaviate for images: {str(e)}")
            
            else:  # By Similarity to Image
                uploaded_search_image = st.file_uploader("Upload an image to find similar ones", type=["jpg", "jpeg", "png"])
                
                if uploaded_search_image is not None:
                    search_image = Image.open(uploaded_search_image)
                    st.image(search_image, caption="Query Image", use_column_width=True)
                    
                    image_similarity_limit = st.slider("Number of similar images", min_value=3, max_value=20, value=5)
                    
                    if st.button("Find Similar Images"):
                        with st.spinner("Searching for similar images..."):
                            # Save temporary file
                            temp_path = f"temp_search_image_{int(time.time())}.jpg"
                            search_image.save(temp_path)
                            
                            try:
                                # Generate embedding
                                embedding = get_image_embedding(temp_path)
                                
                                # Search Weaviate for similar images
                                result = st.session_state.weaviate_client.query.get(
                                    "ImageElement", 
                                    ["file_path", "file_name", "source_file", "image_id", "metadata", "_additional {certainty}"]
                                ).with_near_vector({
                                    "vector": embedding
                                }).with_limit(image_similarity_limit).do()
                                
                                if "data" in result and "Get" in result["data"] and "ImageElement" in result["data"]["Get"]:
                                    results = result["data"]["Get"]["ImageElement"]
                                    
                                    # Display results
                                    st.subheader(f"Found {len(results)} similar images")
                                    
                                    cols = st.columns(3)
                                    for i, item in enumerate(results):
                                        with cols[i % 3]:
                                            # Show relevance score
                                            if "_additional" in item and "certainty" in item["_additional"]:
                                                st.progress(item["_additional"]["certainty"])
                                                st.caption(f"Similarity: {item['_additional']['certainty']:.2%}")
                                            
                                            # Show image details
                                            st.caption(f"File: {item['file_name']}")
                                            st.caption(f"Source: {item['source_file']}")
                                            
                                            # Try to display the actual image if available
                                            try:
                                                if "file_path" in item and os.path.exists(item["file_path"]):
                                                    img = Image.open(item["file_path"])
                                                    st.image(img, caption=item['file_name'], use_column_width=True)
                                                else:
                                                    st.info("Image file not accessible for preview")
                                            except Exception:
                                                st.info("Image preview not available")
                                else:
                                    st.warning("No similar images found")
                            except Exception as e:
                                st.error(f"Error searching for similar images: {str(e)}")
                            
                            # Clean up temp file
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
        
        else:  # Combined Search
            combined_query = st.text_input("Enter your search query", "drone camera setup")
            combined_limit = st.slider("Number of results per type", min_value=3, max_value=10, value=3)
            
            if st.button("Combined Search"):
                with st.spinner("Performing combined search..."):
                    # Generate embedding for the query
                    if st.session_state.text_model is None:
                        st.session_state.text_model = SentenceTransformer("all-MiniLM-L6-v2")
                    
                    query_embedding = st.session_state.text_model.encode(combined_query).tolist()
                    
                    try:
                        # Search for text
                        text_result = st.session_state.weaviate_client.query.get(
                            "DocumentElement", 
                            ["text", "type", "source_file", "element_id", "is_chunk", "chunk_index", 
                             "total_chunks", "metadata", "_additional {certainty}"]
                        ).with_near_vector({
                            "vector": query_embedding
                        }).with_limit(combined_limit).do()
                        
                        # Search for images
                        image_result = st.session_state.weaviate_client.query.get(
                            "ImageElement", 
                            ["file_path", "file_name", "source_file", "image_id", "metadata", "_additional {certainty}"]
                        ).with_near_vector({
                            "vector": query_embedding
                        }).with_limit(combined_limit).do()
                        
                        # Process results
                        text_results = []
                        if "data" in text_result and "Get" in text_result["data"] and "DocumentElement" in text_result["data"]["Get"]:
                            text_results = text_result["data"]["Get"]["DocumentElement"]
                        
                        image_results = []
                        if "data" in image_result and "Get" in image_result["data"] and "ImageElement" in image_result["data"]["Get"]:
                            image_results = image_result["data"]["Get"]["ImageElement"]
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader(f"Found {len(text_results)} text results")
                            
                            for i, item in enumerate(text_results):
                                with st.expander(f"Result {i+1}: {item['type']} - {item['source_file']}"):
                                    # Show relevance score
                                    if "_additional" in item and "certainty" in item["_additional"]:
                                        st.progress(item["_additional"]["certainty"])
                                        st.caption(f"Relevance: {item['_additional']['certainty']:.2%}")
                                    
                                    # Show text content
                                    st.markdown(item['text'][:300] + "..." if len(item['text']) > 300 else item['text'])
                        
                        with col2:
                            st.subheader(f"Found {len(image_results)} image results")
                            
                            for i, item in enumerate(image_results):
                                # Show relevance score
                                if "_additional" in item and "certainty" in item["_additional"]:
                                    st.progress(item["_additional"]["certainty"])
                                    st.caption(f"Relevance: {item['_additional']['certainty']:.2%}")
                                
                                # Show image details
                                st.caption(f"File: {item['file_name']}")
                                st.caption(f"Source: {item['source_file']}")
                                
                                # Try to display the actual image if available
                                try:
                                    if "file_path" in item and os.path.exists(item["file_path"]):
                                        img = Image.open(item["file_path"])
                                        st.image(img, caption=item['file_name'], use_column_width=True)
                                    else:
                                        st.info("Image file not accessible for preview")
                                except Exception:
                                    st.info("Image preview not available")
                    
                    except Exception as e:
                        st.error(f"Error performing combined search: {str(e)}")

# Add requirements.txt file
with open('requirements_streamlit.txt', 'w') as f:
    f.write('''
streamlit>=1.28.0
numpy>=1.23.0
pandas>=1.5.0
plotly>=5.18.0
scikit-learn>=1.3.0
umap-learn>=0.5.4
sentence-transformers>=2.2.2
weaviate-client>=3.24.1
Pillow>=10.0.0
matplotlib>=3.7.0
''')