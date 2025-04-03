import os
from typing import Optional

from mcp.server.fastmcp import Context
from unstructured_client.models.operations import (
    CreateDestinationRequest,
    DeleteDestinationRequest,
    GetDestinationRequest,
    UpdateDestinationRequest,
)
from unstructured_client.models.shared import (
    CreateDestinationConnector,
    LocalDestinationConnectorConfigInput,
    UpdateDestinationConnector,
)

from connectors.utils import create_log_for_created_updated_connector


def _prepare_local_config(
    directory_path: str,
) -> LocalDestinationConnectorConfigInput:
    """Prepare the Local directory destination connector configuration."""
    return LocalDestinationConnectorConfigInput(
        directory_path=directory_path,
    )


async def create_local_destination(
    ctx: Context,
    name: str,
    directory_path: str,
) -> str:
    """Create a local directory destination connector.

    Args:
        name: A unique name for this connector
        directory_path: The local directory path where files will be stored

    Returns:
        String containing the created destination connector information
    """
    client = ctx.request_context.lifespan_context.client
    config = _prepare_local_config(directory_path)

    destination_connector = CreateDestinationConnector(
        name=name,
        type="local",
        config=config,
    )

    try:
        response = await client.destinations.create_destination_async(
            request=CreateDestinationRequest(
                create_destination_connector=destination_connector,
            ),
        )
        result = create_log_for_created_updated_connector(
            response,
            connector_name="Local Directory",
            connector_type="Destination",
            created_or_updated="Created",
        )
        return result
    except Exception as e:
        return f"Error creating local directory destination connector: {str(e)}"


async def update_local_destination(
    ctx: Context,
    destination_id: str,
    directory_path: Optional[str] = None,
) -> str:
    """Update a local directory destination connector.

    Args:
        destination_id: ID of the destination connector to update
        directory_path: The local directory path where files will be stored

    Returns:
        String containing the updated destination connector information
    """
    client = ctx.request_context.lifespan_context.client

    # Get the current destination connector configuration
    try:
        get_response = await client.destinations.get_destination_async(
            request=GetDestinationRequest(destination_id=destination_id),
        )
        current_config = get_response.destination_connector_information.config
    except Exception as e:
        return f"Error retrieving destination connector: {str(e)}"

    # Update configuration with new values
    config = dict(current_config)

    if directory_path is not None:
        config["directory_path"] = directory_path

    destination_connector = UpdateDestinationConnector(config=config)

    try:
        response = await client.destinations.update_destination_async(
            request=UpdateDestinationRequest(
                destination_id=destination_id,
                update_destination_connector=destination_connector,
            ),
        )
        result = create_log_for_created_updated_connector(
            response,
            connector_name="Local Directory",
            connector_type="Destination",
            created_or_updated="Updated",
        )
        return result
    except Exception as e:
        return f"Error updating local directory destination connector: {str(e)}"


async def delete_local_destination(ctx: Context, destination_id: str) -> str:
    """Delete a local directory destination connector.

    Args:
        destination_id: ID of the destination connector to delete

    Returns:
        String containing the result of the deletion
    """
    client = ctx.request_context.lifespan_context.client

    try:
        _ = await client.destinations.delete_destination_async(
            request=DeleteDestinationRequest(destination_id=destination_id),
        )
        return f"Local Directory Destination Connector with ID {destination_id} deleted successfully"
    except Exception as e:
        return f"Error deleting local directory destination connector: {str(e)}"