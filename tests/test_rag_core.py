import tempfile
import unittest
from pathlib import Path

from core.rag.chunker import TextChunker
from core.rag.corpus import (
    DEFAULT_CORPUS_DIR,
    get_default_corpus_dir,
    load_document_from_file,
    load_synthetic_corpus,
)
from core.rag.document import Chunk, Document, RetrievedChunk
from core.rag.retriever import LocalRetriever


class TestDocumentRepresentation(unittest.TestCase):
    """Tests covering Document, Chunk, and RetrievedChunk representations."""

    def test_document_creation_and_attributes(self):
        doc = Document(
            document_id="doc-001",
            title="Safety Guidelines",
            source="policies/safety.json",
            content="Always wear protective eye gear in the laboratory.",
            metadata={"department": "safety", "level": 1},
        )
        self.assertEqual(doc.document_id, "doc-001")
        self.assertEqual(doc.title, "Safety Guidelines")
        self.assertEqual(doc.source, "policies/safety.json")
        self.assertEqual(doc.content, "Always wear protective eye gear in the laboratory.")
        self.assertEqual(doc.metadata["department"], "safety")
        self.assertEqual(doc.metadata["level"], 1)

    def test_document_to_dict_and_from_dict(self):
        doc = Document(
            document_id="doc-002",
            title="Clean Desk Policy",
            source="policies/clean_desk.json",
            content="Lock screens before stepping away.",
            metadata={"classification": "internal"},
        )
        d = doc.to_dict()
        self.assertEqual(d["document_id"], "doc-002")
        self.assertEqual(d["title"], "Clean Desk Policy")
        self.assertEqual(d["source"], "policies/clean_desk.json")
        self.assertEqual(d["content"], "Lock screens before stepping away.")
        self.assertEqual(d["metadata"], {"classification": "internal"})

        reconstructed = Document.from_dict(d)
        self.assertEqual(reconstructed.document_id, doc.document_id)
        self.assertEqual(reconstructed.title, doc.title)
        self.assertEqual(reconstructed.source, doc.source)
        self.assertEqual(reconstructed.content, doc.content)
        self.assertEqual(reconstructed.metadata, doc.metadata)

    def test_chunk_creation_and_attributes(self):
        chunk = Chunk(
            document_id="doc-001",
            chunk_id="doc-001#chunk-0",
            text="Wear eye gear.",
            metadata={"chunk_index": 0},
        )
        self.assertEqual(chunk.document_id, "doc-001")
        self.assertEqual(chunk.chunk_id, "doc-001#chunk-0")
        self.assertEqual(chunk.text, "Wear eye gear.")
        self.assertEqual(chunk.metadata["chunk_index"], 0)

        d = chunk.to_dict()
        self.assertEqual(d["chunk_id"], "doc-001#chunk-0")
        reconstructed = Chunk.from_dict(d)
        self.assertEqual(reconstructed.chunk_id, chunk.chunk_id)
        self.assertEqual(reconstructed.text, chunk.text)

    def test_retrieved_chunk_score_and_inheritance(self):
        retrieved = RetrievedChunk(
            document_id="doc-001",
            chunk_id="doc-001#chunk-0",
            text="Wear eye gear.",
            metadata={"tag": "lab"},
            score=1.452,
        )
        self.assertIsInstance(retrieved, Chunk)
        self.assertEqual(retrieved.score, 1.452)
        d = retrieved.to_dict()
        self.assertEqual(d["score"], 1.452)

        from_c = RetrievedChunk.from_chunk(retrieved, score=2.5)
        self.assertEqual(from_c.score, 2.5)
        self.assertEqual(from_c.chunk_id, "doc-001#chunk-0")


class TestTextChunker(unittest.TestCase):
    """Tests covering deterministic text chunking."""

    def test_chunking_short_document_produces_single_chunk(self):
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        doc = Document(
            document_id="short-doc",
            title="Short Note",
            source="notes.txt",
            content="Short content that fits easily.",
            metadata={"author": "alice"},
        )
        chunks = chunker.chunk_document(doc)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "short-doc#chunk-0")
        self.assertEqual(chunks[0].text, doc.content)
        self.assertEqual(chunks[0].metadata["author"], "alice")
        self.assertEqual(chunks[0].metadata["title"], "Short Note")
        self.assertEqual(chunks[0].metadata["source"], "notes.txt")
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)

    def test_chunking_long_document_with_overlap(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz"
        doc = Document(
            document_id="long-doc",
            title="Alphabet Code",
            source="code.txt",
            content=text,
        )
        chunks = chunker.chunk_document(doc)
        self.assertGreater(len(chunks), 1)
        # Step = 50 - 10 = 40
        self.assertEqual(chunks[0].chunk_id, "long-doc#chunk-0")
        self.assertEqual(chunks[0].text, text[:50])
        self.assertEqual(chunks[1].chunk_id, "long-doc#chunk-1")
        self.assertEqual(chunks[1].text, text[40:62])

    def test_deterministic_chunking_consistency(self):
        chunker = TextChunker(chunk_size=30, chunk_overlap=5)
        text = "Deterministic testing is essential for reproducible security evaluations."
        chunks_1 = chunker.chunk_text(text, document_id="det-doc")
        chunks_2 = chunker.chunk_text(text, document_id="det-doc")

        self.assertEqual(len(chunks_1), len(chunks_2))
        for c1, c2 in zip(chunks_1, chunks_2):
            self.assertEqual(c1.chunk_id, c2.chunk_id)
            self.assertEqual(c1.text, c2.text)
            self.assertEqual(c1.metadata, c2.metadata)

    def test_empty_or_whitespace_content(self):
        chunker = TextChunker()
        self.assertEqual(chunker.chunk_text(""), [])
        self.assertEqual(chunker.chunk_text("   \n\t  "), [])

    def test_invalid_parameters_raise_error(self):
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=0)
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=-10)
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=-1)
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=150)


class TestLocalRetriever(unittest.TestCase):
    """Tests covering deterministic local retrieval."""

    def setUp(self):
        self.doc_it = Document(
            document_id="doc-it",
            title="IT Hardware",
            source="it.json",
            content="Laptops are replaced every three years according to the refresh schedule.",
            metadata={"dept": "it"},
        )
        self.doc_hr = Document(
            document_id="doc-hr",
            title="HR Vacation",
            source="hr.json",
            content="Employees receive twenty days of annual leave and vacation time.",
            metadata={"dept": "hr"},
        )
        self.doc_facility = Document(
            document_id="doc-facility",
            title="Visitor Badges",
            source="facility.json",
            content="Visitors must obtain guest badges at the front reception desk.",
            metadata={"dept": "facility"},
        )
        self.retriever = LocalRetriever.from_documents(
            documents=[self.doc_it, self.doc_hr, self.doc_facility],
            top_k=2,
        )

    def test_deterministic_retrieval_ranking_and_scores(self):
        query = "laptop refresh schedule"
        results_1 = self.retriever.retrieve(query)
        results_2 = self.retriever.retrieve(query)

        self.assertEqual(len(results_1), len(results_2))
        self.assertGreater(len(results_1), 0)
        for r1, r2 in zip(results_1, results_2):
            self.assertEqual(r1.chunk_id, r2.chunk_id)
            self.assertEqual(r1.score, r2.score)

    def test_relevance_matching(self):
        query = "laptop refresh schedule"
        results = self.retriever.retrieve(query)
        self.assertGreaterEqual(len(results), 1)
        top_match = results[0]
        self.assertEqual(top_match.document_id, "doc-it")
        self.assertGreater(top_match.score, 0.0)

    def test_top_k_behavior(self):
        # Default top_k is 2
        res_default = self.retriever.retrieve("laptop vacation visitor")
        self.assertEqual(len(res_default), 2)

        # Override top_k to 1
        res_k1 = self.retriever.retrieve("laptop vacation visitor", top_k=1)
        self.assertEqual(len(res_k1), 1)

        # Override top_k to 3
        res_k3 = self.retriever.retrieve("laptop vacation visitor", top_k=3)
        self.assertEqual(len(res_k3), 3)

        # Override top_k to 0 or negative
        self.assertEqual(self.retriever.retrieve("laptop", top_k=0), [])
        self.assertEqual(self.retriever.retrieve("laptop", top_k=-1), [])

    def test_empty_retrieval(self):
        # Empty corpus
        empty_retriever = LocalRetriever()
        self.assertEqual(empty_retriever.retrieve("any query"), [])

        # Empty or whitespace query
        self.assertEqual(self.retriever.retrieve(""), [])
        self.assertEqual(self.retriever.retrieve("   "), [])

    def test_unknown_query_returns_empty_list(self):
        # Terms that do not exist anywhere in the corpus
        unknown_query = "quantum zebra submarine astrophysics"
        results = self.retriever.retrieve(unknown_query)
        self.assertEqual(results, [])

    def test_metadata_preservation_in_retrieved_chunks(self):
        results = self.retriever.retrieve("visitor guest badges")
        self.assertGreater(len(results), 0)
        chunk = results[0]
        self.assertEqual(chunk.document_id, "doc-facility")
        self.assertIn("dept", chunk.metadata)
        self.assertEqual(chunk.metadata["dept"], "facility")
        self.assertEqual(chunk.metadata["title"], "Visitor Badges")
        self.assertEqual(chunk.metadata["source"], "facility.json")


class TestCorpusLoader(unittest.TestCase):
    """Tests covering safe synthetic corpus loading and path containment."""

    def test_default_corpus_dir_exists(self):
        corpus_dir = get_default_corpus_dir()
        self.assertTrue(corpus_dir.exists())
        self.assertTrue(corpus_dir.is_dir())

    def test_load_synthetic_corpus(self):
        docs = load_synthetic_corpus()
        self.assertGreaterEqual(len(docs), 5)
        doc_ids = {d.document_id for d in docs}
        expected_ids = {
            "doc-it-support",
            "doc-remote-work",
            "doc-travel-expense",
            "doc-visitor-access",
            "doc-annual-leave",
        }
        self.assertTrue(expected_ids.issubset(doc_ids))

        for doc in docs:
            self.assertTrue(doc.document_id)
            self.assertTrue(doc.title)
            self.assertTrue(doc.source)
            self.assertTrue(doc.content)
            self.assertIsInstance(doc.metadata, dict)

    def test_path_traversal_outside_corpus_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir) / "evil.json"
            temp_file.write_text('{"document_id": "bad", "title": "bad", "content": "bad"}')

            with self.assertRaises(ValueError) as ctx:
                load_document_from_file(temp_file, allowed_dir=DEFAULT_CORPUS_DIR)
            self.assertIn("outside the allowed corpus directory", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
