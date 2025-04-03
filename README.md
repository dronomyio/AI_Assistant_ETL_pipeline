# AI Agent ETL Pipeline

This project contains two ETL pipelines that use the Unstructured API for processing documents:

1. **LocalETL**: Processes local documentation files and stores the processed data in a local directory
2. **MCPHackathon**: Processes research papers from Google Drive and stores the data in MongoDB

> **Note**: Streamlit visualization has been removed from this repository due to compatibility issues. To run the ETL pipeline, please use the Docker-based approach described below.

## Dockerized Setup

### Prerequisites
- Docker and Docker Compose installed
- Unstructured API key from [Unstructured platform](https://unstructured.io/)
- (Optional for MCPHackathon) MongoDB connection string and Google Drive service account

### Setup

1. **Configure Environment Variables**

   Copy the example .env file and update with your credentials:
   ```
   cp .env.example .env
   ```
   
   Edit the .env file to add your Unstructured API key and other required credentials.

2. **Build and Run the Docker Containers**

   To start the LocalETL service:
   ```
   docker-compose up local-etl
   ```

   To start both services (if you have MongoDB and Google Drive credentials):
   ```
   docker-compose up
   ```

3. **Access the Services**

   - LocalETL: http://localhost:8080
   - MCPHackathon (if enabled): http://localhost:8081

### Using the Client

To interact with the server, you can use the minimal client provided in the project:

```bash
# For LocalETL
python LocalETL/minimal_client/client.py "http://localhost:8080/sse"

# For MCPHackathon
python MCPHackathon/minimal_client/client.py "http://localhost:8081/sse"
```

### Claude Desktop Integration

1. Edit your Claude Desktop configuration:
   ```bash
   # For macOS or Linux:
   code ~/Library/Application\ Support/Claude/claude_desktop_config.json

   # For Windows:
   code $env:AppData\Claude\claude_desktop_config.json
   ```

2. Add the dockerized MCP server configuration:
   ```json
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

3. Restart Claude Desktop.

## Project Structure

- **LocalETL/** - Local file processing ETL pipeline
- **MCPHackathon/** - Research paper processing ETL pipeline
- **data/** - Source data directory
- **processed_data/** - Output directory for processed data

## Observability with Arize Phoenix

This project includes integration with [Arize Phoenix](https://github.com/Arize-ai/phoenix), an open-source observability tool for LLM applications.

### Setting Up Phoenix

#### Docker Setup (Recommended)

The easiest way to use Phoenix is with Docker Compose:

1. Build and run the Docker containers:
   ```bash
   # Stop any running containers and rebuild
   docker-compose -f docker-compose-phoenix.yml down
   # Start in detached mode
   docker-compose -f docker-compose-phoenix.yml up --build -d
   ```

2. Open your browser to http://localhost:6006 to view the Phoenix dashboard.

##### Troubleshooting

If you can't access the Phoenix UI:

- Verify the containers are running: `docker ps | grep phoenix`
- Check container logs: `docker logs ai_agent_etl_pipeline-etl-phoenix-1`
- Try accessing with a different browser or in incognito mode
- If using Docker Desktop, ensure port forwarding is working correctly
- Try explicitly setting the port in your browser: http://127.0.0.1:6006 instead of localhost

This will:
- Start a Phoenix UI server container on port 6006
- Run the ETL processor container to process files and send telemetry data to Phoenix
- Mount your local data and processed_data directories to the containers

#### Local Setup (Alternative)

If you prefer to run Phoenix locally without Docker:

1. Install the required dependencies:
   ```bash
   pip install -r requirements_phoenix.txt
   ```

2. Run the ETL pipeline with Phoenix enabled:
   ```bash
   python bulk_process_files_with_phoenix.py
   ```

3. To view the Phoenix UI:
   ```bash
   # In a separate terminal
   python -m phoenix.server.main serve
   ```
   
   Then open your browser to http://localhost:6006 to view the Phoenix dashboard.
   
   > **Note**: If you have issues accessing the UI on the default port (6006), try using environment variables: `PHOENIX_HOST=127.0.0.1 PHOENIX_PORT=8765 python -m phoenix.server.main serve`

#### Docker Setup

1. Build and run using Docker Compose:
   ```bash
   docker-compose -f docker-compose-phoenix.yml up
   ```

   This will start both:
   - The ETL processor container that processes your files
   - The Phoenix UI container for visualizing the telemetry data

2. View the Phoenix dashboard at http://localhost:6006

### Features

Phoenix observability provides insights into your ETL pipeline:

- **Trace visualization**: See the entire ETL process flow in a hierarchical view
- **Span details**: Examine each step of the pipeline (extraction, transformation, loading)
- **LLM metrics**: Monitor embedding generation performance
- **Error tracking**: Identify and debug issues in the pipeline
- **Performance analytics**: Analyze processing times and bottlenecks

### Component Integration

Phoenix traces the following components:
- File extraction processes
- Text chunking and transformation
- Embedding generation for text and images
- Vector database operations
- Search queries and results

### Example Usage

After processing your documents with Phoenix enabled, you can also perform semantic searches with observability:

```bash
# Search for specific content in processed documents
python query_with_phoenix.py "search query here"

# Specify number of results to return
python query_with_phoenix.py --top-k 10 "search query here"

# Disable Phoenix telemetry for the search
python query_with_phoenix.py --disable-phoenix "search query here"
```

This will search through processed documents and send search telemetry to Phoenix for visualization.

## Additional Resources

For more detailed information on each service:

- [LocalETL README](LocalETL/README.md)
- [MCPHackathon README](MCPHackathon/README.md)
- [Arize Phoenix Documentation](https://docs.arize.com/phoenix/)