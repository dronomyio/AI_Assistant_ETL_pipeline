.PHONY: debug
debug:
	mcp dev local_mcp/server.py

.PHONY: sse-client
sse-client:
	uv run python minimal_client/client.py "http://127.0.0.1:8080/sse"

.PHONY: sse-server
sse-server:
	uv run python local_mcp/server.py --host 127.0.0.1 --port 8080

.PHONY: test
test:
	uv run pytest tests/

.PHONY: install-pre-commit
install-pre-commit:
	uv pip install ".[dev]"