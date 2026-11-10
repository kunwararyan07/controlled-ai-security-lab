import copy
from typing import Any, Dict, List, Optional, Sequence

from core.rag.document import Chunk, Document


class TextChunker:
    """
    Deterministic text chunker for splitting documents into chunks.

    Splits text using a sliding window strategy with configurable chunk size
    and chunk overlap. Preserves all parent document metadata and enriches
    chunks with chunk index and source information.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """
        Initialize the TextChunker.

        Args:
            chunk_size: Maximum character length for each chunk (must be > 0).
            chunk_overlap: Number of overlapping characters between adjacent chunks (must be >= 0 and < chunk_size).
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be greater than 0, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        text: str,
        document_id: str = "doc",
        base_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Deterministically split raw text into Chunk instances.

        Args:
            text: Text content to split.
            document_id: Identifier of the parent document.
            base_metadata: Optional base metadata dictionary to propagate to chunks.

        Returns:
            List of deterministic Chunk objects.
        """
        if not text or not text.strip():
            return []

        base_meta = copy.deepcopy(base_metadata) if base_metadata is not None else {}
        text_len = len(text)

        # Single chunk case
        if text_len <= self.chunk_size:
            meta = copy.deepcopy(base_meta)
            meta["chunk_index"] = 0
            chunk = Chunk(
                document_id=document_id,
                chunk_id=f"{document_id}#chunk-0",
                text=text,
                metadata=meta,
            )
            return [chunk]

        # Multi-chunk sliding window
        step = self.chunk_size - self.chunk_overlap
        chunks: List[Chunk] = []
        start = 0
        idx = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end]
            meta = copy.deepcopy(base_meta)
            meta["chunk_index"] = idx

            chunk = Chunk(
                document_id=document_id,
                chunk_id=f"{document_id}#chunk-{idx}",
                text=chunk_text,
                metadata=meta,
            )
            chunks.append(chunk)
            idx += 1

            if end >= text_len:
                break
            start += step

        return chunks

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Split a Document into deterministic Chunks, preserving and enriching metadata.

        Args:
            document: The Document to split.

        Returns:
            List of Chunk instances with preserved document metadata.
        """
        base_meta = copy.deepcopy(document.metadata)
        if document.title and "title" not in base_meta:
            base_meta["title"] = document.title
        if document.source and "source" not in base_meta:
            base_meta["source"] = document.source

        return self.chunk_text(
            text=document.content,
            document_id=document.document_id,
            base_metadata=base_meta,
        )

    def chunk_documents(self, documents: Sequence[Document]) -> List[Chunk]:
        """
        Split a sequence of Documents into deterministic Chunks.

        Args:
            documents: Sequence of Document instances.

        Returns:
            Flattened list of all generated Chunk instances.
        """
        all_chunks: List[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
