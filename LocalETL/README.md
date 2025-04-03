# Local Documentation ETL with Unstructured API

This project creates an ETL pipeline for processing local documentation files using the Unstructured API. The data is extracted from local files, transformed using Unstructured API capabilities, and loaded to a local directory for further use.

## Table of Contents:
1. [Setup](#setup)
2. [Requirements](#requirements)
3. [Project Flow](#project-flow)
4. [Available Tools](#available-tools)
5. [Using the ETL Pipeline](#using-the-etl-pipeline)
6. [Claude Desktop Integration](#claude-desktop-integration)
7. [Debugging Tools](#debugging-tools)
8. [Running Locally](#running-locally)

## Setup
Install dependencies:
- `uv add "mcp[cli]"`
- `uv pip install --upgrade unstructured-client python-dotenv`

or use `uv sync`.

## Requirements

Before you can begin working with the **Local ETL** project, make sure you have the following setup:

1. **UNSTRUCTURED_API_KEY**  
   - Get your API key from the [Unstructured platform](https://unstructured.io/) to access their API for document processing.

2. **.env.template**  
   - The `.env.template` file includes all the required environment variables. Copy this file to `.env` and set the necessary values for the keys mentioned above.

   Example `.env` file:
   ```bash
   UNSTRUCTURED_API_KEY="<key-here>"
   ```

## Project Flow

1. User Query to MCP Client

2. Claude Interacts with `LOCAL_MCP` Server
   - Claude forwards the user's query to the custom MCP server.

3. MCP Tool Executes Unstructured API
   - `LOCAL_MCP` interacts with the Unstructured API to process the local documentation files, extract relevant information, and convert it into structured JSON data.

4. Structured Data (JSON) Output is stored in a local directory
   - The result from the Unstructured API is transformed into JSON format, which can be further utilized.

## Available Tools

| Tool | Description |
|------|-------------|
| `list_sources` | Lists available sources from the Unstructured API. |
| `get_source_info` | Get detailed information about a specific source connector. |
| `create_local_source` | Create a local directory source connector. |
| `update_local_source` | Update an existing local source connector by params. |
| `delete_local_source` | Delete a source connector by source id. |
| `list_destinations` | Lists available destinations from the Unstructured API. |
| `get_destination_info` | Get detailed info about a specific destination connector. |
| `create_local_destination` | Create a local directory destination connector by params. |
| `update_local_destination` | Update an existing local destination connector by destination id. |
| `delete_local_destination` | Delete a local directory destination connector by destination id. |
| `list_workflows` | Lists workflows from the Unstructured API. |
| `get_workflow_info` | Get detailed information about a specific workflow. |
| `create_workflow` | Create a new workflow with source, destination id, etc. |
| `run_workflow` | Run a specific workflow with workflow id |
| `update_workflow` | Update an existing workflow by params. |
| `delete_workflow` | Delete a specific workflow by id. |
| `list_jobs` | Lists jobs for a specific workflow from the Unstructured API. |
| `get_job_info` | Get detailed information about a specific job by job id. |
| `cancel_job` |Delete a specific job by id. |

## Using the ETL Pipeline

### 1. **Set Up Required Connectors**

#### Input Local Source Connector:
- **Create a Local Source Connector** to access your local documentation directory.
- **Test the connection** to ensure accessibility.

#### Output Local Destination Connector:
- **Set up a Local Destination Connector** to store processed data in a target directory.
- **Test the connection** to ensure accessibility.

### 2. **Develop the Workflow**

1. **Define Connectors**: Set up the **Local Directory** source and destination connectors.
   
2. **Partitioning**: Use **Auto partitioning** for optimal document splitting.

3. **Chunking**: Apply appropriate chunking strategies for manageable text segments.

4. **Enrichment**: Use **NER** to extract entities and other enrichment options.

5. **Embedding**: Convert text into embeddings for querying or analysis.

Note: Adjust any step (partitioning, chunking, enrichment, embedding) as needed.

### 3. **Set Up Claude Desktop**

1. Install **Claude Desktop** and integrate it with the LOCAL_MCP server.
2. **Restart Claude** to link with the MCP server and ensure workflow functionality.

### 4. **Query and Run the Workflow**

- Use **Claude** to interact with the system and execute queries to list, create, edit, delete and run the workflow.

## Claude Desktop Integration

To install in Claude Desktop:

1. Go to `claude_desktop_config.json` by running the below command.

```bash
# For macOS or Linux:
code ~/Library/Application\ Support/Claude/claude_desktop_config.json

# For Windows:
code $env:AppData\Claude\claude_desktop_config.json
```

2. In that file add:
```bash
{
    "mcpServers":
    {
        "LOCAL_MCP":
        {
            "command": "ABSOLUTE/PATH/TO/.local/bin/uv",
            "args":
            [
                "--directory",
                "ABSOLUTE/PATH/TO/YOUR-LOCAL-ETL-REPO/local_mcp",
                "run",
                "server.py"
            ],
            "env":
            [
            "UNSTRUCTURED_API_KEY":"<your key>"
            ],
            "disabled": false
        }
    }
}
```
3. Restart Claude Desktop.

## Debugging tools

Anthropic provides `MCP Inspector` tool to debug/test your MCP server. Run the following command to spin up a debugging UI. From there, you will be able to add environment variables (pointing to your local env) on the left pane. Include your personal API key there as env var. Go to `tools`, you can test out the capabilities you add to the MCP server.
```
mcp dev local_mcp/server.py
```

If you need to log request call parameters to `UnstructuredClient`, set the environment variable `DEBUG_API_REQUESTS=false`.
The logs are stored in a file with the format `unstructured-client-{date}.log`, which can be examined to debug request call parameters to `UnstructuredClient` functions.

## Running Locally

```
# in one terminal, run the server:
uv run python local_mcp/server.py --host 127.0.0.1 --port 8080

or
make sse-server

# in another terminal, run the client:
uv run python minimal_client/client.py "http://127.0.0.1:8080/sse"
or
make sse-client
```

Hint: `ctrl+c` out of the client first, then the server. Otherwise the server appears to hang.