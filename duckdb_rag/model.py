import torch
import logging
from typing import Any
from transformers import AutoModel, AutoTokenizer


def load_model(model_name: str = "pfnet/plamo-embedding-1b") -> tuple[Any, Any]:
    """モデルとトークナイザーをロードする

    Args:
        model_name: 使用するモデル名

    Returns:
        Tuple[Any, Any]: (model, tokenizer)
    """
    logging.info(f"Loading model and tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device}")
        model = model.to(device)

        return model, tokenizer
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise


def encode_document(model: Any, tokenizer: Any, documents: list[str]) -> torch.Tensor:
    """ドキュメントをベクトル化する

    Args:
        model: 埋め込みモデル
        tokenizer: トークナイザー
        documents: エンコードするドキュメントのリスト

    Returns:
        torch.Tensor: エンコードされたドキュメントベクトル
    """
    with torch.inference_mode():
        return model.encode_document(documents, tokenizer)


def encode_query(model: Any, tokenizer: Any, query: str) -> torch.Tensor:
    """検索クエリをベクトル化する

    Args:
        model: 埋め込みモデル
        tokenizer: トークナイザー
        query: 検索クエリテキスト

    Returns:
        torch.Tensor: エンコードされたクエリベクトル
    """
    with torch.inference_mode():
        return model.encode_query(query, tokenizer)


def get_device_info() -> dict:
    """現在のデバイス情報を取得する

    Returns:
        dict: デバイス情報の辞書
    """
    device_info = {}

    if torch.cuda.is_available():
        device_info["device"] = f"cuda ({torch.cuda.get_device_name()})"
        device_info["gpu_memory_usage"] = (
            f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
        )
    else:
        device_info["device"] = "cpu"

    return device_info
