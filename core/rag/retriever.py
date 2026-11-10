import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence

from core.rag.chunker import TextChunker
from core.rag.document import Chunk, Document, RetrievedChunk


class LocalRetriever:
    """
    Deterministic local retriever for searching a controlled in-memory corpus.

    Implements pure-Python BM25 lexical relevance scoring with deterministic
    tie-breaking. Operates entirely in memory and never performs network access
    or arbitrary filesystem operations.
    """

    def __init__(
        self,
        corpus: Optional[Sequence[Document]] = None,
        chunks: Optional[Sequence[Chunk]] = None,
        chunker: Optional[TextChunker] = None,
        top_k: int = 3,
        min_score: float = 0.0,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """
        Initialize the LocalRetriever.

        Args:
            corpus: Optional sequence of Documents to index (will be chunked).
            chunks: Optional pre-chunked Chunk instances to index.
            chunker: Optional TextChunker for chunking Documents (defaults to TextChunker()).
            top_k: Default number of top scoring chunks to return.
            min_score: Minimum relevance score threshold (chunks at or below this are excluded).
            k1: BM25 term frequency saturation parameter.
            b: BM25 document length normalization parameter.
        """
        self.top_k = top_k
        self.min_score = min_score
        self.k1 = k1
        self.b = b
        self._chunker = chunker or TextChunker()

        self._documents: List[Document] = []
        self._chunks: List[Chunk] = []

        # BM25 index structures
        self._chunk_tokens: List[List[str]] = []
        self._chunk_tfs: List[Counter] = []
        self._chunk_lengths: List[int] = []
        self._df: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._avgdl: float = 0.0

        if corpus:
            self.add_documents(corpus)
        elif chunks:
            self.add_chunks(chunks)

    @property
    def chunks(self) -> List[Chunk]:
        """Return the list of indexed chunks."""
        return list(self._chunks)

    @property
    def documents(self) -> List[Document]:
        """Return the list of indexed documents."""
        return list(self._documents)

    @staticmethod
    def _stem(token: str) -> str:
        """Lightweight deterministic English plural suffix reduction."""
        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
        return token

    def _tokenize(self, text: str) -> List[str]:
        """
        Deterministic word tokenization with light stemming.

        Extracts lowercase alphanumeric tokens and normalizes common plurals.
        """
        if not text:
            return []
        raw_tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
        return [self._stem(t) for t in raw_tokens]

    def _rebuild_index(self) -> None:
        """Recompute BM25 inverted index and statistics across all indexed chunks."""
        n = len(self._chunks)
        self._chunk_tokens = [self._tokenize(c.text) for c in self._chunks]
        self._chunk_tfs = [Counter(tokens) for tokens in self._chunk_tokens]
        self._chunk_lengths = [len(tokens) for tokens in self._chunk_tokens]

        if n == 0:
            self._df = {}
            self._idf = {}
            self._avgdl = 0.0
            return

        total_length = sum(self._chunk_lengths)
        self._avgdl = total_length / n

        # Document frequency
        df: Dict[str, int] = {}
        for tf in self._chunk_tfs:
            for term in tf.keys():
                df[term] = df.get(term, 0) + 1
        self._df = df

        # Inverted document frequency (Robertson / BM25 formula)
        idf: Dict[str, float] = {}
        for term, freq in df.items():
            # Standard BM25 IDF with smoothing
            idf[term] = math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
        self._idf = idf

    def add_documents(self, documents: Sequence[Document]) -> None:
        """
        Add Documents to the retriever, chunking them with the configured chunker.

        Args:
            documents: Sequence of Document instances to index.
        """
        for doc in documents:
            self._documents.append(doc)
            new_chunks = self._chunker.chunk_document(doc)
            self._chunks.extend(new_chunks)
        self._rebuild_index()

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        """
        Add pre-chunked Chunk instances directly to the retriever.

        Args:
            chunks: Sequence of Chunk instances to index.
        """
        self._chunks.extend(chunks)
        self._rebuild_index()

    def clear(self) -> None:
        """Clear all indexed documents, chunks, and index structures."""
        self._documents.clear()
        self._chunks.clear()
        self._chunk_tokens.clear()
        self._chunk_tfs.clear()
        self._chunk_lengths.clear()
        self._df.clear()
        self._idf.clear()
        self._avgdl = 0.0

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """
        Retrieve relevant chunks from the controlled corpus for a user query.

        Args:
            query: The user query string.
            top_k: Optional override for the number of top scoring chunks to return.

        Returns:
            List of RetrievedChunk instances ordered by score descending, with deterministic tie-breaking.
        """
        if not isinstance(query, str) or not query.strip():
            return []

        effective_top_k = top_k if top_k is not None else self.top_k
        if effective_top_k <= 0:
            return []

        if not self._chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        query_terms = set(query_tokens)
        candidates: List[RetrievedChunk] = []

        avgdl = self._avgdl if self._avgdl > 0.0 else 1.0

        for idx, chunk in enumerate(self._chunks):
            tf_map = self._chunk_tfs[idx]
            dl = self._chunk_lengths[idx]

            score = 0.0
            for term in query_terms:
                if term in tf_map:
                    tf = tf_map[term]
                    idf = self._idf.get(term, 0.0)
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
                    score += idf * (numerator / denominator)

            if score > self.min_score:
                candidates.append(
                    RetrievedChunk.from_chunk(chunk=chunk, score=round(score, 6))
                )

        # Deterministic sorting: highest score first, then by document_id and chunk_id
        candidates.sort(key=lambda c: (-c.score, c.document_id, c.chunk_id))

        return candidates[:effective_top_k]

    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Alias for retrieve()."""
        return self.retrieve(query=query, top_k=top_k)

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[Document],
        chunker: Optional[TextChunker] = None,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> "LocalRetriever":
        """
        Factory method to instantiate a LocalRetriever directly from a sequence of Documents.

        Args:
            documents: Sequence of Document instances.
            chunker: Optional TextChunker.
            top_k: Default top_k.
            min_score: Minimum score cutoff.

        Returns:
            A populated LocalRetriever instance.
        """
        retriever = cls(chunker=chunker, top_k=top_k, min_score=min_score)
        retriever.add_documents(documents)
        return retriever
