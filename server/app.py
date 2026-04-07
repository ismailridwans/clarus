"""Entry point for uv run server / openenv serve.

Exposes the FastAPI app and a main() function for the [project.scripts]
entry point defined in pyproject.toml.
"""

from server.main import app  # re-export for ASGI runners


def main() -> None:
    """Launch the Clarus server with uvicorn (used by `uv run server`)."""
    import uvicorn

    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )


if __name__ == "__main__":
    main()
