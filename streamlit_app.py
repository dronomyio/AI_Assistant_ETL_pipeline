#!/usr/bin/env python3
"""
AI ETL Pipeline Visualizer - Entry Point
This is a simple redirector to the cloud-compatible version.
"""
import streamlit as st

# Create Streamlit app UI
st.title("AI ETL Pipeline Visualizer")
st.markdown("""
## Please use the cloud-compatible version

We've detected you're using the main app entry point, but we recommend using the cloud-compatible version directly.

Please run:
```bash
streamlit run streamlit_app_cloud.py
```

Or update your Streamlit Cloud configuration to use `streamlit_app_cloud.py` as the main file path with `requirements_cloud.txt`.
""")

st.info("The cloud-compatible version has fewer dependencies and works reliably on Streamlit Cloud.")

st.markdown("""
### Quick Setup Instructions for Streamlit Cloud:

1. Go to your app's settings
2. Change the Main file path to: `streamlit_app_cloud.py`
3. Change the Requirements file to: `requirements_cloud.txt`
4. Save and relaunch your app
""")

# Also show error indication
st.error("This entry point may have dependency issues on Streamlit Cloud. Please use the cloud-compatible version instead.")