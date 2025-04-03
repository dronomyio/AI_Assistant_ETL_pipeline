import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx


def setup_logger() -> logging.Logger:
    """Set up a logger for debugging API requests."""
    today = datetime.now().strftime("%Y-%m-%d")
    logger = logging.getLogger("unstructured-client")
    logger.setLevel(logging.DEBUG)

    # Create a file handler for logging
    file_handler = logging.FileHandler(f"unstructured-client-{today}.log")
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)

    return logger


class CustomHttpClient:
    """
    A custom HTTP client wrapper for debugging API requests.
    Logs the request body to a file for debugging purposes.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        """Initialize the custom HTTP client with an async client."""
        self.client = client
        self.logger = setup_logger()

    def sanitize_auth_header(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Sanitize headers to avoid logging sensitive information."""
        sanitized = headers.copy()
        if "authorization" in sanitized:
            sanitized["authorization"] = "Bearer [REDACTED]"
        if "x-api-key" in sanitized:
            sanitized["x-api-key"] = "[REDACTED]"
        return sanitized

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        content: Optional[Union[str, bytes]] = None,
        json: Optional[Any] = None,
    ) -> httpx.Response:
        """
        Make an HTTP request and log the details.
        """
        # Log the request details
        sanitized_headers = self.sanitize_auth_header(headers or {})
        request_log = {
            "method": method,
            "url": url,
            "params": params,
            "headers": sanitized_headers,
        }

        if content:
            request_log["content"] = (
                content.decode("utf-8") if isinstance(content, bytes) else content
            )
        if json:
            request_log["json"] = json

        self.logger.debug(f"Making request: {request_log}")

        # Make the actual request
        response = await self.client.request(
            method,
            url,
            params=params,
            headers=headers,
            cookies=cookies,
            content=content,
            json=json,
        )

        # Log the response summary
        response_log = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }

        try:
            response_log["body"] = response.json()
        except Exception:
            response_log["body"] = response.text[:1000] + (
                "..." if len(response.text) > 1000 else ""
            )

        self.logger.debug(f"Response received: {response_log}")

        return response