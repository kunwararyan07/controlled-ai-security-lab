import json
import os
from pathlib import Path
from typing import List, Optional, Union

from core.rag.document import Document

# Default path relative to repository root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CORPUS_DIR = REPO_ROOT / "synthetic_data" / "rag" / "documents"


def get_default_corpus_dir() -> Path:
    """Return the absolute path to the default synthetic RAG corpus directory."""
    return DEFAULT_CORPUS_DIR


def load_document_from_file(
    file_path: Union[str, Path],
    allowed_dir: Optional[Union[str, Path]] = None,
) -> Document:
    """
    Safely load a Document from a JSON file with strict path confinement.

    Args:
        file_path: Path to the JSON document file.
        allowed_dir: Permitted directory root. Access outside this directory is rejected.
                     Defaults to the default synthetic corpus directory.

    Returns:
        A Document instance loaded from the file.

    Raises:
        ValueError: If file_path resolves outside allowed_dir, or has an invalid format.
        FileNotFoundError: If the specified file does not exist.
    """
    base_dir = Path(allowed_dir or DEFAULT_CORPUS_DIR).resolve()
    target_path = Path(file_path).resolve()

    # Security check: strictly enforce that target_path is within allowed_dir
    try:
        target_path.relative_to(base_dir)
    except ValueError:
        raise ValueError(
            f"Security violation: path '{file_path}' resolves outside the allowed corpus directory '{base_dir}'."
        )

    if not target_path.exists():
        raise FileNotFoundError(f"Document file not found: {target_path}")

    if not target_path.is_file():
        raise ValueError(f"Path is not a regular file: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Document JSON must contain an object, got {type(data).__name__}")

    required_keys = {"document_id", "title", "content"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Document is missing required keys: {sorted(missing)}")

    return Document.from_dict(data)


def load_synthetic_corpus(
    corpus_dir: Optional[Union[str, Path]] = None,
) -> List[Document]:
    """
    Load all synthetic Documents from the controlled corpus directory in deterministic order.

    Args:
        corpus_dir: Optional path to the directory containing synthetic documents.
                    Defaults to DEFAULT_CORPUS_DIR.

    Returns:
        List of Document instances ordered by filename.
    """
    target_dir = Path(corpus_dir or DEFAULT_CORPUS_DIR).resolve()

    if not target_dir.exists() or not target_dir.is_dir():
        return []

    # Deterministic alphabetical ordering of documents
    doc_files = sorted([f for f in os.listdir(target_dir) if f.endswith(".json")])

    documents: List[Document] = []
    for fname in doc_files:
        full_path = target_dir / fname
        doc = load_document_from_file(full_path, allowed_dir=target_dir)
        documents.append(doc)

    return documents
