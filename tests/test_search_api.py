import pytest
from unittest.mock import MagicMock
import torch

from server import search_documents, Document
from tests.conftest import MockContext


@pytest.fixture
def mock_setup():
    # Mockオブジェクトのセットアップ
    mock_model = MagicMock()
    mock_embedding = torch.tensor([[0.1] * 2048])
    mock_model.encode_query.return_value = mock_embedding

    mock_tokenizer = MagicMock()

    mock_result = MagicMock()
    mock_result.fetchall.return_value = [
        ("テストドキュメント1", 0.1),
        ("テストドキュメント2", 0.2),
        ("テストドキュメント3", 0.3),
    ]

    mock_conn = MagicMock()
    mock_conn.sql.return_value = mock_result

    # コンテキスト生成
    mock_ctx = MockContext(model=mock_model, tokenizer=mock_tokenizer, conn=mock_conn)

    yield {
        "ctx": mock_ctx,
        "model": mock_model,
        "tokenizer": mock_tokenizer,
        "conn": mock_conn,
        "embedding": mock_embedding,
    }


@pytest.mark.asyncio
async def test_search_documents_api(mock_setup):
    # search_documents関数を呼び出し
    query = "テスト検索クエリ"
    limit = 3
    results = await search_documents(ctx=mock_setup["ctx"], query=query, limit=limit)

    # モデルでクエリがエンコードされたか確認
    mock_setup["model"].encode_query.assert_called_once_with(
        query, mock_setup["tokenizer"]
    )

    # SQLクエリが正しく実行されたか確認
    mock_setup["conn"].sql.assert_called_once()
    sql_query = mock_setup["conn"].sql.call_args[0][0]
    assert "SELECT content, array_cosine_distance" in sql_query
    assert "ORDER BY distance" in sql_query
    assert "LIMIT ?" in sql_query

    # パラメータが正しいか確認
    assert mock_setup["conn"].sql.call_args[1]["params"]
    params = mock_setup["conn"].sql.call_args[1]["params"]
    assert isinstance(params, list)
    assert len(params) == 2
    assert params[1] == limit

    # 戻り値が期待通りであることを確認
    assert len(results) == 3
    assert isinstance(results[0], Document)
    assert results[0].content == "テストドキュメント1"
    assert results[0].distance == 0.1
    assert results[1].content == "テストドキュメント2"
    assert results[1].distance == 0.2
    assert results[2].content == "テストドキュメント3"
    assert results[2].distance == 0.3


@pytest.mark.asyncio
async def test_search_documents_empty_result(mock_setup):
    # 空の結果をシミュレート
    mock_setup["conn"].sql.return_value.fetchall.return_value = []

    # search_documents関数を呼び出し
    results = await search_documents(ctx=mock_setup["ctx"], query="存在しないクエリ")

    # 空のリストが返されることを確認
    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_documents_default_limit(mock_setup):
    # デフォルトのlimitでsearch_documents関数を呼び出し
    await search_documents(ctx=mock_setup["ctx"], query="テストクエリ")

    # SQLクエリのパラメータでlimitが5になっていることを確認
    params = mock_setup["conn"].sql.call_args[1]["params"]
    assert params[1] == 5
