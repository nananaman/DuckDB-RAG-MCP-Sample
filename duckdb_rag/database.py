import logging
import os
from typing import Any

import duckdb


def initialize_db(home_directory: str | None = None) -> Any:
    """DuckDBデータベースを初期化する

    Args:
        home_directory: DuckDBのホームディレクトリパス（任意）

    Returns:
        duckdb.Connection: 初期化されたデータベース接続
    """
    logging.info("Initializing DuckDB database")
    try:
        conn = duckdb.connect()
        if home_directory:
            conn.sql(f"SET home_directory='{home_directory}'")
        conn.sql("INSTALL vss")
        conn.sql("LOAD vss")

        conn.sql("CREATE SEQUENCE IF NOT EXISTS id_sequence START 1;")
        conn.sql(
            "CREATE TABLE IF NOT EXISTS article (id INTEGER DEFAULT nextval('id_sequence'), content TEXT, vector FLOAT[2048]);"
        )
        logging.info("Database initialized successfully")
        return conn
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
        raise


def load_vectors_from_parquet(conn: Any, parquet_path: str) -> int:
    """Parquetファイルからベクトルをロードする

    Args:
        conn: DuckDB接続
        parquet_path: Parquetファイルのパス

    Returns:
        int: ロードされたドキュメント数
    """
    if os.path.exists(parquet_path):
        logging.info(f"Loading vectors from parquet file '{parquet_path}'")
        try:
            conn.sql(
                f"INSERT INTO article SELECT * FROM read_parquet('{parquet_path}')"
            )
            result = conn.sql("SELECT COUNT(*) FROM article")
            fetch_result = result.fetchone()
            if fetch_result is not None:
                count = int(fetch_result[0])
                logging.info(f"Loaded {count} document vectors from parquet file")
                return count
            else:
                logging.info("No document vectors loaded from parquet file")
                return 0
        except Exception as e:
            logging.error(f"Failed to load vectors from parquet file: {e}")
            raise
    else:
        logging.warning(f"Parquet file '{parquet_path}' not found")
        return 0


def save_vectors_to_parquet(conn: Any, parquet_path: str) -> bool:
    """ベクトルデータをParquetファイルとして保存する

    Args:
        conn: DuckDB接続
        parquet_path: 保存先のParquetファイルパス

    Returns:
        bool: 保存が成功したかどうか
    """
    logging.info(f"Saving vectorized data to '{parquet_path}'")
    try:
        conn.sql(f"COPY article TO '{parquet_path}' (FORMAT PARQUET)")
        logging.info(f"Successfully saved vectors to '{parquet_path}'")
        return True
    except Exception as e:
        logging.error(f"Failed to save vectors to parquet file: {e}")
        return False


def search_documents(
    conn: Any, vector: list[float], limit: int = 5
) -> list[tuple[str, float]]:
    """ベクトル検索を実行する

    Args:
        conn: DuckDB接続
        vector: 検索クエリのベクトル
        limit: 返す結果の最大数

    Returns:
        list[tuple[str, float]]: ドキュメントコンテンツと距離のリスト
    """
    result = conn.sql(
        """
        SELECT content, array_cosine_distance(vector, ?::FLOAT[2048]) as distance
        FROM article
        ORDER BY distance
        LIMIT ?
        """,
        params=[vector, limit],
    )

    return result.fetchall()


def add_document(conn: Any, content: str, vector: list[float]) -> bool:
    """ドキュメントをデータベースに追加する

    Args:
        conn: DuckDB接続
        content: ドキュメントのテキスト内容
        vector: ドキュメントのベクトル表現

    Returns:
        bool: 追加が成功したかどうか
    """
    try:
        conn.execute(
            "INSERT INTO article (content, vector) VALUES (?, ?)",
            [content, vector],
        )
        return True
    except Exception as e:
        logging.error(f"Error adding document: {e}")
        return False


def add_documents_batch(
    conn: Any, contents: list[str], vectors: list[list[float]]
) -> bool:
    """複数のドキュメントをデータベースにバッチで追加する

    Args:
        conn: DuckDB接続
        contents: ドキュメントのテキスト内容のリスト
        vectors: ドキュメントのベクトル表現のリスト（contentsと同じ順序）

    Returns:
        bool: 追加が成功したかどうか
    """
    try:
        # リストをzipして、バッチクエリで一度に挿入
        data = list(zip(contents, vectors))
        conn.executemany(
            "INSERT INTO article (content, vector) VALUES (?, ?)",
            data,
        )
        return True
    except Exception as e:
        logging.error(f"Error adding documents batch: {e}")
        return False


def get_document_count(conn: Any) -> int:
    """データベース内のドキュメント数を取得する

    Args:
        conn: DuckDB接続

    Returns:
        int: ドキュメント数
    """
    try:
        result = conn.sql("SELECT COUNT(*) FROM article")
        fetch_result = result.fetchone()
        if fetch_result is not None:
            return int(fetch_result[0])
        return 0
    except Exception as e:
        logging.error(f"Error fetching document count: {e}")
        return -1
