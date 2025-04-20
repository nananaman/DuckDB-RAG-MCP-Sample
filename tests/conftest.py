import pytest
from unittest.mock import patch, MagicMock


# モックコンテキスト
class MockContext:
    def __init__(self, model=None, tokenizer=None, conn=None):
        self.request_context = MagicMock()
        self.request_context.lifespan_context = MagicMock()
        self.request_context.lifespan_context.model = model
        self.request_context.lifespan_context.tokenizer = tokenizer
        self.request_context.lifespan_context.conn = conn


# FastMCP のモックを用意
@pytest.fixture(autouse=True)
def mock_fastmcp_decorator():
    """
    FastMCPのデコレータをモックする
    """
    with patch("server.mcp") as mock_mcp:
        # lifespan デコレータのモック
        def mock_lifespan():
            def decorator(func):
                async def wrapper():
                    return await func()

                return wrapper

            return decorator

        # tool デコレータのモック
        def mock_tool():
            def decorator(func):
                async def wrapper(*args, **kwargs):
                    return await func(*args, **kwargs)

                return wrapper

            return decorator

        mock_mcp.lifespan = mock_lifespan
        mock_mcp.tool = mock_tool

        yield mock_mcp
