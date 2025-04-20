import glob
import logging
import os


def get_markdown_files(directory_path: str) -> list[str]:
    """指定されたディレクトリからすべてのMarkdownファイルのパスを取得する

    Args:
        directory_path: Markdownファイルを探すディレクトリパス

    Returns:
        str | None: Markdownファイルのパスリスト
    """
    return glob.glob(os.path.join(directory_path, "*.md"))


def load_markdown_file(file_path: str) -> str | None:
    """単一のMarkdownファイルを読み込み、内容を返す

    Args:
        file_path: ファイルパス

    Returns:
        str | None: ファイルの内容、エラー時はNone
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            logging.info(f"Loaded: {os.path.basename(file_path)}")
            return content
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None


def configure_logging(level: int = logging.INFO) -> None:
    """ロギング設定を構成する

    Args:
        level: ロギングレベル
    """
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def get_file_info(file_path: str) -> dict:
    """ファイル情報を取得する

    Args:
        file_path: ファイルパス

    Returns:
        dict: ファイル情報
    """
    file_info: dict = {}

    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        file_info["exists"] = True
        file_info["size"] = int(file_size)
        file_info["size_mb"] = f"{file_size / 1e6:.1f} MB"
    else:
        file_info["exists"] = False

    return file_info
