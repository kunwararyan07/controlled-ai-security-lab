from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.rag.document import RetrievedChunk


@dataclass
class RAGAgentResult:
    """
    Structured result of a RAG Agent execution.

    Preserves at minimum:
    - user_query: The raw query issued by the user.
    - retrieved_chunks: The chunks retrieved from the corpus.
    - model_response: Raw response string from the ModelAdapter.
    - final_response: Processed response returned to the user.
    - error: Error message if an error occurred during execution.
    """

    user_query: str
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    model_response: Optional[str] = None
    final_response: Optional[str] = None
    error: Optional[str] = None
    session_id: Optional[str] = None
    retrieval_query: Optional[str] = None
    prompt: Optional[str] = None
    duration_seconds: Optional[float] = None

    @property
    def user_input(self) -> str:
        """Alias for user_query for compatibility with ToolUsingAgent conventions."""
        return self.user_query

    @property
    def chunk_count(self) -> int:
        """Return the number of retrieved chunks."""
        return len(self.retrieved_chunks)

    @property
    def has_error(self) -> bool:
        """Return True if an execution error occurred."""
        return self.error is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary representation."""
        return {
            "user_query": self.user_query,
            "user_input": self.user_input,
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks],
            "model_response": self.model_response,
            "final_response": self.final_response,
            "error": self.error,
            "session_id": self.session_id,
            "retrieval_query": self.retrieval_query,
            "prompt": self.prompt,
            "duration_seconds": self.duration_seconds,
            "chunk_count": self.chunk_count,
            "has_error": self.has_error,
        }
