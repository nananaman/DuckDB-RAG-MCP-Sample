# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Communication Guidelines
日本語で話すようにしてください

## Commands
- Run application: `uv run main.py <args>`
- FastMCP server: `uv run mcp dev server.py`
- Formatting: `mise run fmt`
- Linting: `mise run lint`
- Testing: `mise run test`

## Code Style Guidelines
- **Naming**: Use snake_case for variables, functions, and modules
- **Typing**: Include type hints for all function parameters and return values
- **Imports**: Group imports as: stdlib, third-party, local modules
- **Docstrings**: Use Google-style docstrings for all functions
- **Error handling**: Use specific exception types with descriptive messages
- **Resource management**: Ensure proper cleanup of database connections
- **Testing**: Include tests for new functionality
- **Formatting**: Follow PEP 8 conventions with 88-character line limit

## Project Structure
- Main functionality in `main.py` (CLI tool)
- Server implementation in `server.py` (FastMCP API)
- Vector data stored in `vectors.parquet`
