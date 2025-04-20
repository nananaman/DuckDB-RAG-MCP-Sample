import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, cast

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel

import duckdb_rag as dr


class Document(BaseModel):
    content: str
    distance: float = 0.0


@dataclass
class AppContext:
    model: Any
    tokenizer: Any
    conn: Any


# アプリケーションのライフサイクル管理
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    # ロギング設定
    dr.configure_logging()
    logging.info("Server initialization starting")

    try:
        # モデル初期化
        model, tokenizer = dr.load_model()

        # DuckDB初期化
        conn = dr.initialize_db(home_directory="/tmp")

        # Parquetファイル読み込み
        parquet_path = os.environ.get("VECTOR_PARQUET", "vectors.parquet")
        dr.load_vectors_from_parquet(conn, parquet_path)

        logging.info("Server initialization completed successfully")
    except Exception as e:
        logging.error(f"Server initialization failed: {e}")
        raise

    try:
        # AppContextインスタンスを返す
        yield AppContext(model=model, tokenizer=tokenizer, conn=conn)
    finally:
        # クリーンアップ処理
        logging.info("Server shutdown initiated")
        if conn:
            try:
                conn.close()
                logging.info("Database connection closed")
            except Exception as e:
                logging.error(f"Error closing database connection: {e}")
        logging.info("Server shutdown completed")


# サーバーインスタンス
mcp = FastMCP(
    "DuckDB-RAG-MCP-Sample",
    dependencies=["duckdb", "torch", "transformers", "sentencepiece"],
    lifespan=app_lifespan,
)


# 検索API
@mcp.tool()
async def search_documents(ctx: Context, query: str, limit: int = 5) -> list[Document]:
    """
    Search for documents that match the query.
    """
    logging.info(f"Searching documents with query: '{query}', limit: {limit}")

    try:
        # コンテキスト経由でリソースへアクセス
        app_ctx = ctx.request_context.lifespan_context
        model = app_ctx.model
        tokenizer = app_ctx.tokenizer
        conn = app_ctx.conn

        # クエリエンベディング生成と検索
        query_embedding = dr.encode_query(model, tokenizer, query)
        query_vector = query_embedding.cpu().squeeze().numpy().tolist()
        result_rows = dr.search_documents(conn, cast(list[float], query_vector), limit)

        # 結果変換
        documents = []
        for row in result_rows:
            documents.append(Document(content=row[0], distance=float(row[1])))

        logging.info(f"Found {len(documents)} matching documents")
        return documents
    except Exception as e:
        logging.error(f"Error searching documents: {e}")
        raise


# システム状態確認API
@mcp.tool()
async def get_system_status(ctx: Context) -> dict:
    """
    Get the current system status.
    """
    logging.info("Getting system status")

    try:
        # コンテキスト経由でリソースへアクセス
        app_ctx = ctx.request_context.lifespan_context
        conn = app_ctx.conn

        status: dict[str, object] = {
            "model_name": "pfnet/plamo-embedding-1b",
            "model_status": "initialized",
            "vector_db_status": "connected",
        }

        # デバイス情報
        device_info = dr.get_device_info()
        status.update(device_info)

        # Parquetファイル情報
        parquet_path = os.environ.get("VECTOR_PARQUET", "vectors.parquet")
        file_info = dr.get_file_info(parquet_path)

        if file_info["exists"]:
            status["vector_file"] = f"{parquet_path} ({file_info['size_mb']})"
        else:
            status["vector_file"] = f"{parquet_path} (not found)"

        # ドキュメント数
        doc_count = dr.get_document_count(conn)
        if doc_count >= 0:
            status["document_count"] = doc_count
        else:
            status["document_count"] = "Error: Could not retrieve count"

        return status
    except Exception as e:
        logging.error(f"Error getting system status: {e}")
        raise


if __name__ == "__main__":
    mcp.run()
