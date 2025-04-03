#!/usr/bin/env python3
"""
Redirector to the minimal Streamlit app for compatibility with Streamlit Cloud.
This version is designed to have minimal dependencies to avoid import errors.
"""
import streamlit as st
import os
import sys

# Try to import the minimal app directly
try:
    # Add the current directory to the path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    # Import functions from the minimal app
    from streamlit_app_minimal import *
    
    # Print success message
    print("Successfully imported minimal app")
    
except ImportError as e:
    st.error(f"Failed to import minimal app: {e}")
    
    # Basic app content as fallback
    st.title("AI ETL Pipeline Visualizer")
    
    st.markdown("""
    ## Welcome to the AI ETL Pipeline Visualizer
    
    This app is designed to visualize and interact with the ETL pipeline with vector embeddings, 
    but we're currently experiencing import errors.
    
    Please try the minimal version instead:
    
    ```bash
    streamlit run streamlit_app_minimal.py
    ```
    
    Or check the repository for updated versions.
    """)
    
    st.error("""
    We encountered an error loading all dependencies. Please check that all required packages are installed:
    
    ```
    pip install -r requirements_streamlit.txt
    ```
    
    If you're using Streamlit Cloud, please use the `streamlit_app_minimal.py` file instead.
    """)