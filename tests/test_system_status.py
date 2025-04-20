import pytest
from unittest.mock import patch, MagicMock, Mock

from server import get_system_status
from tests.conftest import MockContext


@pytest.fixture
def mock_setup():
    # Mockオブジェクトのセットアップ
    mock_conn = MagicMock()
    mock_conn_sql = MagicMock()
    mock_conn_sql.fetchone.return_value = [10]
    mock_conn.sql.return_value = mock_conn_sql

    # コンテキスト生成
    mock_ctx = MockContext(conn=mock_conn)

    # torchのモック - システムでtorch.cudaが利用可能なことを示す
    mock_torch = Mock()
    mock_torch.cuda = Mock()
    mock_torch.cuda.is_available = Mock(return_value=True)
    mock_torch.cuda.get_device_name = Mock(return_value="Test GPU")
    mock_torch.cuda.memory_allocated = Mock(return_value=1000000000)  # 1GB

    # OSモックの作成
    mock_os = MagicMock()
    mock_os.environ.get.return_value = "vectors.parquet"
    mock_os.path.exists.return_value = True
    mock_os.path.getsize.return_value = 5000000  # 5MB

    # モック関数を作成
    mock_get_document_count = MagicMock(return_value=10)
    mock_get_device_info = MagicMock(
        return_value={"device": "cuda (Test GPU)", "gpu_memory_usage": "1.00 GB"}
    )
    mock_get_file_info = MagicMock(return_value={"exists": True, "size_mb": "5.0 MB"})

    with (
        patch("duckdb_rag.get_document_count", mock_get_document_count),
        patch("duckdb_rag.get_device_info", mock_get_device_info),
        patch("duckdb_rag.get_file_info", mock_get_file_info),
        patch("os.environ.get", mock_os.environ.get),
        patch.dict("sys.modules", {"torch": mock_torch}),
    ):
        yield {
            "ctx": mock_ctx,
            "conn": mock_conn,
            "torch": mock_torch,
            "os": mock_os,
            "get_document_count": mock_get_document_count,
            "get_device_info": mock_get_device_info,
            "get_file_info": mock_get_file_info,
        }


@pytest.mark.asyncio
async def test_get_system_status(mock_setup):
    # get_system_status関数を呼び出し
    status = await get_system_status(ctx=mock_setup["ctx"])

    # 結果が辞書であることを確認
    assert isinstance(status, dict)

    # 必要なキーが含まれていることを確認
    assert "document_count" in status
    assert "model_name" in status
    assert "device" in status
    assert "vector_db_status" in status
    assert "vector_file" in status

    # 値が期待通りであることを確認
    assert status["document_count"] == 10
    assert "pfnet/plamo-embedding-1b" in status["model_name"]
    assert status["device"] == "cuda (Test GPU)"
    assert status["vector_db_status"] == "connected"
    assert "vectors.parquet" in status["vector_file"]
    assert "5.0 MB" in status["vector_file"]


@pytest.mark.asyncio
async def test_get_system_status_no_cuda(mock_setup):
    # CUDA非対応環境をシミュレート
    mock_setup["get_device_info"].return_value = {"device": "cpu"}

    # get_system_status関数を呼び出し
    status = await get_system_status(ctx=mock_setup["ctx"])

    # デバイスがCPUになっていることを確認
    assert status["device"] == "cpu"
