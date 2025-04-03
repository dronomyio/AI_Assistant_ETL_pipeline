#!/usr/bin/env python3
"""
ETL Concept Viewer - A conceptual overview of the AI Agent ETL Pipeline
This is a pure informational Streamlit app with no dependencies on the actual codebase.
"""
import streamlit as st

# Set page config
st.set_page_config(
    page_title="AI ETL Pipeline - Concept Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
with st.sidebar:
    st.title("AI Agent ETL Pipeline")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["Overview", "Architecture", "ETL Process", "Vector Search", "Integration"]
    )
    
    # About section
    st.markdown("---")
    st.markdown("""
    **About This App**
    
    This app provides a conceptual overview of the AI Agent ETL Pipeline.
    It does not implement any functionality but serves as educational material.
    
    For the actual implementation, please use the Docker-based approach in the repository.
    """)

# Main content based on selected page
if page == "Overview":
    st.title("AI Agent ETL Pipeline: Conceptual Overview")
    
    st.markdown("""
    ## What is an AI Agent ETL Pipeline?
    
    The AI Agent ETL (Extract, Transform, Load) Pipeline is a system designed to process documents 
    and create structured, searchable data that can be used by AI systems like Claude.
    
    ### Key Components:
    
    1. **Extract**: Gather documents from various sources (local files, Google Drive, etc.)
    
    2. **Transform**: Process documents to extract structured information
       - Break documents into semantic chunks
       - Generate vector embeddings for text and images
       - Create metadata for improved searchability
    
    3. **Load**: Store the processed data in a destination
       - Local file system
       - Vector database (Weaviate)
       - MongoDB
    
    ### Implementation Variants:
    
    This project contains two ETL pipeline variants:
    
    - **LocalETL**: Processes local documentation files and stores locally
    - **MCPHackathon**: Processes Google Drive files and stores in MongoDB
    """)
    
    col1, col2, col3 = st.columns([1,2,1])
    
    with col2:
        st.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*HFlYgB6gVLc4x9R3a2iDcQ.png", 
                caption="Conceptual ETL Pipeline Flow")
    
    st.markdown("""
    ### Benefits of This Approach:
    
    - **Structure Unstructured Data**: Convert raw documents into structured elements
    - **Semantic Understanding**: Create embeddings that capture meaning, not just keywords
    - **Context Preservation**: Maintain document context with overlapping chunks
    - **Multimodal Processing**: Handle both text and images for comprehensive understanding
    - **AI-Ready Output**: Generate data that's optimized for AI agent consumption
    """)

elif page == "Architecture":
    st.title("Architecture: How It All Fits Together")
    
    st.markdown("""
    ## System Architecture
    
    The AI Agent ETL Pipeline is built with a modular, extensible architecture:
    
    ### Core Components
    
    1. **Source Connectors**
       - Local Directory: Process files from a local filesystem
       - Google Drive: Access and process files from Google Drive
       - Extensible design for adding new sources
    
    2. **Processing Pipeline**
       - Document Parsing: Extract text and structure from documents
       - Contextual Chunking: Split documents while preserving semantic meaning
       - Embedding Generation: Create vector representations of content
    
    3. **Destination Connectors**
       - Local Directory: Save processed data locally
       - MongoDB: Store in a MongoDB database
       - Weaviate: Save in a vector database for semantic search
    
    4. **MCP Integration**
       - Model Context Protocol (MCP) server implementation
       - Allows Claude to access data via the pipeline
       - Enables seamless document retrieval during conversations
    """)
    
    st.markdown("""
    ## Technology Stack
    
    The system leverages several key technologies:
    
    - **Python**: Core implementation language
    - **Unstructured API**: Document parsing and element extraction
    - **Docker**: Containerization for consistent deployment
    - **Sentence Transformers**: Vector embedding generation
    - **Weaviate**: Vector database for semantic search
    - **MongoDB**: Document storage for structured data
    - **SSE (Server-Sent Events)**: Communication protocol for streaming results
    """)
    
    st.markdown("""
    ## Data Flow
    
    ```
    [Source] → [Extract] → [Parse Documents] → [Chunk Text] → [Generate Embeddings] → [Load] → [Destination]
    ```
    
    A parallel flow for image processing:
    
    ```
    [Source] → [Extract] → [Process Images] → [Generate Image Embeddings] → [Load] → [Destination]
    ```
    """)

elif page == "ETL Process":
    st.title("ETL Process: Extract, Transform, Load")
    
    tab1, tab2, tab3 = st.tabs(["Extract", "Transform", "Load"])
    
    with tab1:
        st.header("Extract: Gathering Documents")
        
        st.markdown("""
        The extraction phase retrieves documents from various sources:
        
        ### Local Files
        
        ```python
        # Conceptual code for local file extraction
        def extract_local_files(input_dir, recursive=True):
            file_paths = []
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
                if not recursive:
                    break
            return file_paths
        ```
        
        ### Google Drive
        
        ```python
        # Conceptual code for Google Drive extraction
        def extract_gdrive_files(folder_id, mime_types=None):
            service = build_drive_service()
            query = f"'{folder_id}' in parents and trashed=false"
            if mime_types:
                query += f" and mimeType in {mime_types}"
            files = service.files().list(q=query).execute().get('files', [])
            return files
        ```
        
        ### Handling Different File Types
        
        The system can process various file types:
        - Text files (.txt, .md, .csv)
        - Documents (.pdf, .docx)
        - Images (.jpg, .png, etc.)
        
        Each file type requires specific handling during extraction.
        """)
    
    with tab2:
        st.header("Transform: Processing Documents")
        
        st.markdown("""
        The transformation phase processes the raw documents into structured data:
        
        ### Text Processing
        
        1. **Document Parsing**
           - Extract text content from files
           - Identify document elements (titles, paragraphs, lists)
           - Create structured representations
        
        2. **Contextual Chunking**
           - Split long documents into manageable chunks
           - Maintain context through overlapping sections
           - Prefer natural boundaries (paragraphs, sections)
        
        ```python
        # Conceptual code for contextual chunking
        def contextual_chunker(text, chunk_size=1000, chunk_overlap=200):
            chunks = []
            start = 0
            
            while start < len(text):
                # Find end of chunk with natural boundaries
                end = min(start + chunk_size, len(text))
                
                # Try to break at paragraph
                last_break = text.rfind('\\n\\n', start, end)
                if last_break != -1 and last_break > start + chunk_size // 2:
                    end = last_break
                
                # Extract chunk and add to results
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                
                # Move with overlap
                start = end - chunk_overlap if end - chunk_overlap > start else end
            
            return chunks
        ```
        
        3. **Embedding Generation**
           - Convert text chunks to vector embeddings
           - Capture semantic meaning in numerical format
           - Enable similarity-based search
        
        ### Image Processing
        
        1. **Image Analysis**
           - Extract visual features
           - Identify content and context
        
        2. **Image Embedding**
           - Generate vector representations using models like CLIP
           - Enable cross-modal search (find images by text)
        """)
    
    with tab3:
        st.header("Load: Storing Processed Data")
        
        st.markdown("""
        The loading phase stores the processed data in various destinations:
        
        ### Local Storage
        
        ```python
        # Conceptual code for local storage
        def store_locally(elements, output_dir, source_file):
            output_file = os.path.join(output_dir, f"{os.path.basename(source_file)}.json")
            with open(output_file, "w") as f:
                json.dump(elements, f, indent=2)
            return output_file
        ```
        
        ### Vector Database (Weaviate)
        
        ```python
        # Conceptual code for Weaviate storage
        def store_in_weaviate(client, elements, source_file):
            for element in elements:
                client.data_object.create(
                    class_name="DocumentElement",
                    data_object={
                        "text": element["text"],
                        "type": element["type"],
                        "source_file": source_file
                    },
                    vector=element["embedding"]
                )
        ```
        
        ### MongoDB
        
        ```python
        # Conceptual code for MongoDB storage
        def store_in_mongodb(collection, elements, source_file):
            documents = []
            for element in elements:
                doc = element.copy()
                doc["source_file"] = source_file
                documents.append(doc)
            
            collection.insert_many(documents)
        ```
        
        ### Data Organization
        
        The system maintains organizational integrity by:
        - Preserving original file structure
        - Adding metadata about source files
        - Creating unique identifiers for each element
        - Maintaining relationships between chunks of the same document
        """)

elif page == "Vector Search":
    st.title("Vector Search: Finding What You Need")
    
    st.markdown("""
    ## How Vector Search Works
    
    Vector search is a powerful technique for finding semantically similar content:
    
    1. **Embedding Generation**:
       - Convert text/images to numerical vectors
       - Vectors represent semantic meaning in high-dimensional space
       - Similar concepts have vectors that are close together
    
    2. **Similarity Measurement**:
       - Calculate distance between vectors (e.g., cosine similarity)
       - Closer vectors = more similar content
       - Independent of exact wording or syntax
    
    3. **Search Process**:
       - Convert search query to embedding
       - Find closest vectors in the database
       - Return corresponding content
    """)
    
    st.markdown("""
    ## Search Types Supported
    
    The system supports several search types:
    
    ### Text-to-Text Search
    
    ```python
    # Conceptual code for text search
    def semantic_text_search(query, client, limit=5):
        # Generate embedding for query
        query_embedding = get_text_embedding(query)
        
        # Search for similar text elements
        result = client.query.get(
            "DocumentElement", 
            ["text", "type", "source_file"]
        ).with_near_vector({
            "vector": query_embedding
        }).with_limit(limit).do()
        
        return result["data"]["Get"]["DocumentElement"]
    ```
    
    ### Text-to-Image Search
    
    ```python
    # Conceptual code for image search by text
    def semantic_image_search(query, client, limit=5):
        # Generate embedding for query
        query_embedding = get_text_embedding(query)
        
        # Search for similar images
        result = client.query.get(
            "ImageElement", 
            ["file_path", "file_name", "source_file"]
        ).with_near_vector({
            "vector": query_embedding
        }).with_limit(limit).do()
        
        return result["data"]["Get"]["ImageElement"]
    ```
    
    ### Image-to-Image Search
    
    ```python
    # Conceptual code for similar image search
    def find_similar_images(image_path, client, limit=5):
        # Generate embedding for image
        image_embedding = get_image_embedding(image_path)
        
        # Search for similar images
        result = client.query.get(
            "ImageElement", 
            ["file_path", "file_name", "source_file"]
        ).with_near_vector({
            "vector": image_embedding
        }).with_limit(limit).do()
        
        return result["data"]["Get"]["ImageElement"]
    ```
    """)
    
    st.markdown("""
    ## Search Enhancements
    
    The system implements several enhancements to improve search quality:
    
    ### Chunk Recombination
    
    When documents are chunked, search results can be fragmented. 
    Chunk recombination improves user experience:
    
    ```python
    # Conceptual code for chunk recombination
    def recombine_chunks(results):
        # Group chunks by source document
        document_chunks = {}
        for item in results:
            if item["is_chunk"]:
                source = item["source_file"]
                if source not in document_chunks:
                    document_chunks[source] = []
                document_chunks[source].append(item)
        
        # Sort and merge chunks from same document
        final_results = []
        for source, chunks in document_chunks.items():
            chunks.sort(key=lambda x: x["chunk_index"])
            combined_text = " ".join([c["text"] for c in chunks])
            final_results.append({
                "text": combined_text,
                "type": "MergedChunks",
                "source_file": source
            })
        
        return final_results
    ```
    
    ### Cross-Modal Search
    
    The system can combine text and image search for comprehensive results:
    
    ```python
    # Conceptual code for combined search
    def combined_search(query, client, limit=5):
        # Search both text and images
        text_results = semantic_text_search(query, client, limit)
        image_results = semantic_image_search(query, client, limit)
        
        return {
            "text": text_results,
            "images": image_results
        }
    ```
    """)

else:  # Integration
    st.title("Integration: Connecting with Claude")
    
    st.markdown("""
    ## Model Context Protocol (MCP)
    
    The ETL pipeline integrates with Claude using the Model Context Protocol (MCP):
    
    ### What is MCP?
    
    MCP is a protocol developed by Anthropic that allows Claude to:
    - Access external data and tools
    - Retrieve relevant information during conversations
    - Extend its knowledge with domain-specific content
    
    ### How MCP Works with This ETL Pipeline
    
    1. **Setup MCP Server**
       - The pipeline includes MCP server implementations
       - These servers expose processed data to Claude
    
    ```python
    # Conceptual code for MCP server
    async def handle_mcp_request(request):
        # Parse the request from Claude
        query = request.json["query"]
        
        # Perform search using the ETL pipeline
        results = semantic_text_search(query, client, limit=5)
        
        # Return results to Claude
        return {
            "results": results
        }
    ```
    
    2. **Claude Desktop Integration**
       - Configure Claude Desktop with MCP server details
       - Enable Claude to access the processed data
    
    ```json
    // Conceptual Claude Desktop configuration
    {
        "mcpServers":
        {
            "LOCAL_MCP":
            {
                "command": "docker",
                "args": ["exec", "-i", "ai_agent_etl_pipeline-local-etl-1", "python", "local_mcp/server.py"],
                "env": ["UNSTRUCTURED_API_KEY":"<your key>"],
                "disabled": false
            }
        }
    }
    ```
    
    3. **Usage in Conversations**
       - Claude can pull information from your documents
       - Information is retrieved based on semantic relevance
       - User experience is seamless and conversational
    """)
    
    st.markdown("""
    ## API Integration
    
    The system can also be integrated via API endpoints:
    
    ### SSE (Server-Sent Events)
    
    The ETL pipeline uses SSE for streaming results:
    
    ```python
    # Conceptual code for SSE endpoint
    @app.route("/sse")
    async def sse_endpoint():
        # Set up SSE response
        response = await make_sse_response()
        
        # Get query from request
        query = request.args.get("query")
        
        # Process query and stream results
        async for result in process_query(query):
            await response.send(result)
        
        return response
    ```
    
    ### Client Example
    
    ```python
    # Conceptual client code
    def connect_to_sse_server(url, query):
        params = {"query": query}
        response = requests.get(url, params=params, stream=True)
        
        client = SSEClient(response)
        for event in client.events():
            yield json.loads(event.data)
    ```
    
    ### Claude API Integration
    
    The system can also be integrated with Claude API directly:
    
    ```python
    # Conceptual Claude API integration
    def ask_claude_with_context(query, context):
        response = claude_client.messages.create(
            model="claude-3-opus-20240229",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": query}
                ]}
            ],
            system=f"You have access to the following information: {context}"
        )
        return response.content[0].text
    ```
    """)

# Footer
st.markdown("---")
st.caption("AI Agent ETL Pipeline - Concept Viewer")