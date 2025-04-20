import os
import argparse
import logging
import torch

import duckdb_rag as dr


def main():
    # ロギング設定
    dr.configure_logging()

    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="DuckDB RAG ベクトルデータ生成ツール")
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.expanduser("~/Documents/chouge"),
        help="Markdownファイルを読み込むディレクトリパス",
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default="vectors.parquet",
        help="ベクトルを保存するParquetファイルのパス",
    )
    args = parser.parse_args()

    # データベース初期化
    conn = dr.initialize_db()

    # モデルを読み込む
    model, tokenizer = dr.load_model()

    # 指定されたディレクトリからMarkdownファイルのパスを取得
    logging.info(f"Searching for markdown files in '{args.directory}'")
    markdown_files = dr.get_markdown_files(args.directory)
    if not markdown_files:
        logging.warning(f"No markdown files found in directory '{args.directory}'")
        return
    logging.info(f"Found {len(markdown_files)} markdown files")

    # ドキュメントをバッチでベクトル化してDBに保存
    batch_size = 10  # 適切なバッチサイズ
    contents = []

    for file_path in markdown_files:
        doc = dr.load_markdown_file(file_path)
        if doc:
            contents.append(doc)

            # バッチサイズに達したらエンコードして挿入
            if len(contents) >= batch_size:
                with torch.inference_mode():
                    doc_embeddings = dr.encode_document(model, tokenizer, contents)
                    doc_vectors = [
                        embedding.cpu().squeeze().numpy().tolist()
                        for embedding in doc_embeddings
                    ]
                    dr.add_documents_batch(conn, contents, doc_vectors)

                # バッチをクリア
                contents = []

    # 残りのドキュメントを処理
    if contents:
        with torch.inference_mode():
            doc_embeddings = dr.encode_document(model, tokenizer, contents)
            doc_vectors = [
                embedding.cpu().squeeze().numpy().tolist()
                for embedding in doc_embeddings
            ]
            dr.add_documents_batch(conn, contents, doc_vectors)

    # ベクトル化したデータをParquetとして保存
    dr.save_vectors_to_parquet(conn, args.parquet)
    logging.info(f"Vector data saved to '{args.parquet}'")


if __name__ == "__main__":
    main()
