from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Document:
    """
    Representation of a document within the RAG corpus.

    Preserves document identity, title, source origin, textual content,
    and structured metadata.
    """

    document_id: str
    title: str
    source: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert document representation to a dictionary."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "source": self.source,
            "content": self.content,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Instantiate Document from a dictionary."""
        return cls(
            document_id=data["document_id"],
            title=data.get("title", ""),
            source=data.get("source", ""),
            content=data.get("content", ""),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Chunk:
    """
    Representation of an individual chunk extracted from a document.

    Preserves parent document identity, unique chunk identifier, chunk text,
    and associated metadata.
    """

    document_id: str
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk representation to a dictionary."""
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """Instantiate Chunk from a dictionary."""
        return cls(
            document_id=data["document_id"],
            chunk_id=data["chunk_id"],
            text=data.get("text", ""),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class RetrievedChunk(Chunk):
    """
    Representation of a chunk retrieved by the retriever, including relevance score.
    """

    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert retrieved chunk to a dictionary representation."""
        data = super().to_dict()
        data["score"] = self.score
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievedChunk":
        """Instantiate RetrievedChunk from a dictionary."""
        return cls(
            document_id=data["document_id"],
            chunk_id=data["chunk_id"],
            text=data.get("text", ""),
            metadata=dict(data.get("metadata", {})),
            score=float(data.get("score", 0.0)),
        )

    @classmethod
    def from_chunk(cls, chunk: Chunk, score: float = 0.0) -> "RetrievedChunk":
        """Create a RetrievedChunk from an existing Chunk and a relevance score."""
        return cls(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            metadata=dict(chunk.metadata),
            score=score,
        )
