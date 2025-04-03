import os
from typing import Optional, List

from mcp.server.fastmcp import Context
from unstructured_client import UNSET, OptionalNullable
from unstructured_client.models.operations import (
    CreateSourceRequest,
    DeleteSourceRequest,
    GetSourceRequest,
    UpdateSourceRequest,
)
from unstructured_client.models.shared import (
    CreateSourceConnector,
    LocalSourceConnectorConfigInput,
    UpdateSourceConnector,
)

from connectors.utils import (
    create_log_for_created_updated_connector,
)


def _prepare_local_source_config(
    directory_path: str,
    recursive: Optional[bool],
    extensions: OptionalNullable[List[str]] = UNSET,
) -> LocalSourceConnectorConfigInput:
    """Prepare the local directory source connector configuration."""
    return LocalSourceConnectorConfigInput(
        directory_path=directory_path,
        recursive=recursive,
        extensions=extensions,
    )


async def create_local_source(
    ctx: Context,
    name: str,
    directory_path: str,
    recursive: bool = True,
    extensions: OptionalNullable[List[str]] = UNSET,
) -> str:
    """Create a local directory source connector.

    Args:
        name: A unique name for this connector
        directory_path: The path to the local directory
        recursive: Whether to access subfolders within the directory
        extensions: List of file extensions to include

    Returns:
        String containing the created source connector information
    """
    client = ctx.request_context.lifespan_context.client
    config = _prepare_local_source_config(directory_path, recursive, extensions)
    source_connector = CreateSourceConnector(name=name, type="local", config=config)

    try:
        response = await client.sources.create_source_async(
            request=CreateSourceRequest(create_source_connector=source_connector),
        )
        result = create_log_for_created_updated_connector(
            response,
            connector_name="Local Directory",
            connector_type="Source",
            created_or_updated="Created",
        )
        return result
    except Exception as e:
        return f"Error creating local directory source connector: {str(e)}"


async def update_local_source(
    ctx: Context,
    source_id: str,
    directory_path: Optional[str] = None,
    recursive: Optional[bool] = None,
    extensions: OptionalNullable[List[str]] = UNSET,
) -> str:
    """Update a local directory source connector.

    Args:
        source_id: ID of the source connector to update
        directory_path: The path to the local directory
        recursive: Whether to access subfolders within the directory
        extensions: List of file extensions to include

    Returns:
        String containing the updated source connector information
    """
    client = ctx.request_context.lifespan_context.client

    # Get the current source connector configuration
    try:
        get_response = await client.sources.get_source_async(
            request=GetSourceRequest(source_id=source_id),
        )
        current_config = get_response.source_connector_information.config
    except Exception as e:
        return f"Error retrieving source connector: {str(e)}"

    # Update configuration with new values
    config = dict(current_config)

    if directory_path is not None:
        config["directory_path"] = directory_path

    if recursive is not None:
        config["recursive"] = recursive

    if extensions is not UNSET:
        config["extensions"] = extensions

    source_connector = UpdateSourceConnector(config=config)

    try:
        response = await client.sources.update_source_async(
            request=UpdateSourceRequest(
                source_id=source_id,
                update_source_connector=source_connector,
            ),
        )
        result = create_log_for_created_updated_connector(
            response,
            connector_name="Local Directory",
            connector_type="Source",
            created_or_updated="Updated",
        )
        return result
    except Exception as e:
        return f"Error updating local directory source connector: {str(e)}"


async def delete_local_source(ctx: Context, source_id: str) -> str:
    """Delete a local directory source connector.

    Args:
        source_id: ID of the source connector to delete

    Returns:
        String containing the result of the deletion
    """
    client = ctx.request_context.lifespan_context.client

    try:
        _ = await client.sources.delete_source_async(
            request=DeleteSourceRequest(source_id=source_id),
        )
        return f"Local Directory Source Connector with ID {source_id} deleted successfully"
    except Exception as e:
        return f"Error deleting local directory source connector: {str(e)}"