import pytest
from unittest.mock import patch, MagicMock, Mock

from server import app_lifespan


@pytest.fixture
def mock_environment():
    # モックオブジェクトを作成
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_duckdb = MagicMock()
    mock_os = MagicMock()

    # Mockオブジェクトのセットアップ
    mock_conn = MagicMock()
    mock_duckdb.connect.return_value = mock_conn

    mock_tokenizer_instance = MagicMock()
    mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance

    mock_model_instance = MagicMock()
    mock_model.from_pretrained.return_value = mock_model_instance

    # torchのモック
    mock_torch = Mock()
    mock_torch.cuda = Mock()
    mock_torch.cuda.is_available = Mock(return_value=False)

    # OSモックのセットアップ
    mock_os.path.exists.return_value = True
    mock_os.environ.get.return_value = "vectors.parquet"

    # duckdb_ragモジュールのモック
    mock_load_model = MagicMock(
        return_value=(mock_model_instance, mock_tokenizer_instance)
    )
    mock_initialize_db = MagicMock(return_value=mock_conn)
    mock_load_vectors = MagicMock(return_value=10)

    # モジュールを適切にパッチ
    with (
        patch("duckdb_rag.load_model", mock_load_model),
        patch("duckdb_rag.initialize_db", mock_initialize_db),
        patch("duckdb_rag.load_vectors_from_parquet", mock_load_vectors),
        patch("os.path.exists", mock_os.path.exists),
        patch("os.environ.get", mock_os.environ.get),
        patch.dict("sys.modules", {"torch": mock_torch}),
    ):
        yield {
            "torch": mock_torch,
            "duckdb": mock_duckdb,
            "conn": mock_conn,
            "tokenizer_class": mock_tokenizer,
            "tokenizer": mock_tokenizer_instance,
            "model_class": mock_model,
            "model": mock_model_instance,
            "os": mock_os,
            "load_model": mock_load_model,
            "initialize_db": mock_initialize_db,
            "load_vectors": mock_load_vectors,
        }


@pytest.mark.asyncio
async def test_lifespan_initialization(mock_environment):
    # FastMCPのモック
    mock_server = MagicMock()

    # app_lifespanを使ってコンテキストマネージャを生成
    context_manager = app_lifespan(mock_server)

    # コンテキストを開始
    async with context_manager as context:
        # load_modelが呼び出されたことを検証
        mock_environment["load_model"].assert_called_once()

        # initialize_dbが呼び出されたことを検証
        mock_environment["initialize_db"].assert_called_once_with(home_directory="/tmp")

        # load_vectors_from_parquetが呼び出されたことを検証
        mock_environment["load_vectors"].assert_called_once_with(
            mock_environment["conn"], "vectors.parquet"
        )

        # コンテキストが期待通りのプロパティを持っていることを検証
        assert hasattr(context, "model")
        assert hasattr(context, "tokenizer")
        assert hasattr(context, "conn")


@pytest.mark.asyncio
async def test_lifespan_no_parquet_file(mock_environment):
    # Parquetファイルが存在しない状況をシミュレート - load_vectorsを0を返すように設定
    mock_environment["load_vectors"].return_value = 0

    # FastMCPのモック
    mock_server = MagicMock()

    # app_lifespanを使ってコンテキストマネージャを生成
    context_manager = app_lifespan(mock_server)

    # コンテキストを開始
    async with context_manager as context:
        # load_modelが呼び出されたことを検証
        mock_environment["load_model"].assert_called_once()

        # initialize_dbが呼び出されたことを検証
        mock_environment["initialize_db"].assert_called_once_with(home_directory="/tmp")

        # load_vectors_from_parquetが呼び出されたことを検証
        mock_environment["load_vectors"].assert_called_once_with(
            mock_environment["conn"], "vectors.parquet"
        )

        # コンテキストが期待通りのプロパティを持っていることを検証
        assert hasattr(context, "model")
        assert hasattr(context, "tokenizer")
        assert hasattr(context, "conn")
