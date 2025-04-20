from .database import (
    initialize_db,
    load_vectors_from_parquet,
    save_vectors_to_parquet,
    search_documents,
    add_document,
    add_documents_batch,
    get_document_count,
)

from .model import load_model, encode_document, encode_query, get_device_info

from .utils import (
    get_markdown_files,
    load_markdown_file,
    configure_logging,
    get_file_info,
)

__all__ = [
    # database
    "initialize_db",
    "load_vectors_from_parquet",
    "save_vectors_to_parquet",
    "search_documents",
    "add_document",
    "add_documents_batch",
    "get_document_count",
    # model
    "load_model",
    "encode_document",
    "encode_query",
    "get_device_info",
    # utils
    "get_markdown_files",
    "load_markdown_file",
    "configure_logging",
    "get_file_info",
]
