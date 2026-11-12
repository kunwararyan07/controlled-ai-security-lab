#!/usr/bin/env python3
"""
KAS RAG Agent — Interactive Security Console.

Dedicated interactive CLI console for testing and inspecting the RAG Agent,
retrieval ranking, BM25 scores, in-memory synthetic document injection,
and observability events using local Ollama models.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
import uuid

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.rag_agent.agent import RAGAgent
from agents.rag_agent.result import RAGAgentResult
from core.logging.collector import EventCollector
from core.logging.events import Event
from core.models.base import ModelAdapter
from core.models.ollama import OllamaAdapter, OllamaConfigurationError, _validate_model_name
from core.rag.corpus import get_default_corpus_dir, load_synthetic_corpus
from core.rag.document import Document, RetrievedChunk
from core.rag.prompt import RAGPromptBuilder
from core.rag.retriever import LocalRetriever

BANNER = """\
╔════════════════════════════════════════════════════════════╗
║                   KAS RAG AGENT CONSOLE                    ║
╚════════════════════════════════════════════════════════════╝\
"""

HELP_TEXT = """\
Available Commands:
  /help               Show this help message
  /ask <query>        Submit a query to the RAG Agent (or type query directly)
  /docs               List all currently loaded baseline corpus documents
  /inject             Interactively inject an in-memory experimental document
  /injections         List all experimental documents currently injected
  /clear-injections   Remove all injected documents and restore baseline retriever
  /model [name]       Display current model or switch to another local Ollama model
  /topk [number]      Display or change the retrieval top-K value
  /events             Display recent observability events for this session
  /export [prefix]    Export session interaction history to JSON and Markdown
  /quit, /exit        Exit the KAS RAG Agent Console
"""

DEFAULT_MODEL_NAME = "gemma2:2b"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 60.0
DEFAULT_TOP_K = 3
DEFAULT_EXPORT_DIR = REPO_ROOT / "experiments" / "rag" / "manual_results"
MAX_DOCUMENT_ID_LENGTH = 64
MAX_DOCUMENT_TITLE_LENGTH = 200

# Form field label patterns identifying pasted forms or labels
FORM_FIELD_PATTERNS = [
    re.compile(r"(?i)\bdocument\s*id\b(\s*:)?"),
    re.compile(r"(?i)\bdocument\s*title\b(\s*:)?"),
    re.compile(r"(?i)\bdocument\s*content\b(\s*:)?"),
    re.compile(r"(?i)(?<!document)(?<!document\s)\bcontent\s*:"),
    re.compile(r"(?i)(?<!document)(?<!document\s)\btitle\s*:"),
    re.compile(r"(?i)\bclassification\s*:"),
    re.compile(r"(?i)\bdepartment\s*:"),
    re.compile(r"(?i)\bcategory\s*:"),
    re.compile(r"(?i)\bmetadata\s*:"),
    re.compile(r"(?i)\bsource\s*:"),
]


def format_banner() -> str:
    """Return the prominent KAS RAG branding banner."""
    return BANNER


def format_rag_result(result: RAGAgentResult, top_k: int) -> str:
    """
    Format a RAGAgentResult into a structured, readable console report.

    Displays:
    1. Query text
    2. Retrieved document IDs & chunk IDs
    3. Retrieval ranking and BM25 scores
    4. Retrieved chunk content
    5. Final model response
    6. Execution duration and errors (if any)
    """
    lines: List[str] = [
        "",
        "─" * 60,
        "  KAS RAG RETRIEVAL & RESPONSE REPORT",
        "─" * 60,
        f"  Query: {result.user_query}",
        "",
    ]

    if not result.retrieved_chunks:
        lines.append("  Retrieved Chunks: [No relevant context retrieved]")
    else:
        lines.append(f"  Retrieved Chunks ({len(result.retrieved_chunks)} / Top-{top_k}):")
        for rank, chunk in enumerate(result.retrieved_chunks, 1):
            is_injected = (
                chunk.metadata.get("classification") == "synthetic_injected"
                or chunk.metadata.get("is_ephemeral", False)
            )
            origin_tag = "[EXPERIMENTAL INJECTION]" if is_injected else "[BASELINE]"
            title = chunk.metadata.get("title", "Untitled")

            lines.append(f"    [{rank}] Document: {chunk.document_id} {origin_tag}")
            lines.append(f"        Chunk ID:   {chunk.chunk_id}")
            lines.append(f"        BM25 Score: {chunk.score:.6f}")
            lines.append(f"        Title:      {title}")
            lines.append("        Content:")
            for content_line in chunk.text.splitlines():
                lines.append(f"          {content_line}")
            lines.append("")

    # Final response
    if result.final_response is not None and result.final_response.strip():
        lines.append(f"  KAS RAG > {result.final_response.strip()}")
    else:
        lines.append("  KAS RAG > [No final text response]")

    # Timing & errors
    if result.duration_seconds is not None:
        lines.append(f"\n  Duration: {result.duration_seconds:.3f}s")

    if result.error:
        lines.append(f"  Error:    {result.error}")

    lines.append("─" * 60)
    lines.append("")
    return "\n".join(lines)


def validate_document_id(
    raw_input: str,
    existing_ids: Set[str],
    max_length: int = MAX_DOCUMENT_ID_LENGTH,
) -> Tuple[bool, Optional[str], bool]:
    """
    Validate a proposed document ID for ephemeral injection.

    Args:
        raw_input: Raw string entered by the user.
        existing_ids: Set of existing document IDs in baseline and active injections.
        max_length: Maximum allowed length for document ID.

    Returns:
        Tuple of (is_valid, error_message, is_duplicate).
    """
    stripped = raw_input.strip()
    if not stripped:
        return False, "Document ID cannot be empty.", False

    # 1. Detect multiple pasted form fields or embedded form blocks
    matched_labels = [p.pattern for p in FORM_FIELD_PATTERNS if p.search(raw_input)]
    has_newlines = "\n" in raw_input or "\r" in raw_input

    if len(matched_labels) >= 2 or (matched_labels and (has_newlines or ".end" in raw_input or len(stripped) > max_length)):
        return (
            False,
            "Input appears to contain multiple pasted form fields. Please enter only the document identifier.",
            False,
        )

    # 2. Reject IDs containing field labels (e.g. 'Document ID:', 'Document Title:')
    if matched_labels:
        return (
            False,
            "Document ID cannot contain field labels such as 'Document ID:' or 'Document Title:'. Please enter only the document identifier.",
            False,
        )

    # 3. Reject IDs containing newlines
    if has_newlines:
        return (
            False,
            "Document ID cannot contain newline characters. Please enter a single-line identifier.",
            False,
        )

    # 4. Check maximum length
    if len(stripped) > max_length:
        return (
            False,
            f"Document ID exceeds maximum length of {max_length} characters (got {len(stripped)}).",
            False,
        )

    # 5. Check for duplicate IDs
    if stripped in existing_ids:
        return (
            False,
            f"Document ID '{stripped}' already exists.",
            True,
        )

    return True, None, False


def validate_document_title(
    raw_input: str,
    max_length: int = MAX_DOCUMENT_TITLE_LENGTH,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a proposed document title for ephemeral injection.

    Args:
        raw_input: Raw string entered by the user.
        max_length: Maximum allowed length for document title.

    Returns:
        Tuple of (is_valid, error_message).
    """
    stripped = raw_input.strip()
    # Empty title is allowed (falls back to default title)
    if not stripped:
        return True, None

    # 1. Detect multiple pasted form fields or embedded form blocks
    matched_labels = [p.pattern for p in FORM_FIELD_PATTERNS if p.search(raw_input)]
    has_newlines = "\n" in raw_input or "\r" in raw_input

    if len(matched_labels) >= 2 or (matched_labels and (has_newlines or ".end" in raw_input or len(stripped) > max_length)):
        return (
            False,
            "Input appears to contain multiple pasted form fields. Please enter only the document title.",
        )

    # 2. Reject titles containing field labels
    if matched_labels:
        return (
            False,
            "Document Title cannot contain field labels such as 'Document Title:' or 'Title:'. Please enter only the title text.",
        )

    # 3. Reject titles containing newlines
    if has_newlines:
        return (
            False,
            "Document Title cannot contain newline characters. Please enter a single-line title.",
        )

    # 4. Check maximum length
    if len(stripped) > max_length:
        return (
            False,
            f"Document Title exceeds maximum length of {max_length} characters (got {len(stripped)}).",
        )

    return True, None


class InjectionRecord(dict):
    """
    Historical record for an injected synthetic document.

    Preserves complete document data, payload content, structured metadata,
    injection timestamp, and cleanup lifecycle state across the session.
    Subclasses dict for direct JSON serializability while providing
    property-based attribute access.
    """

    def __init__(
        self,
        document: Document,
        status: str = "active",
        injected_at: Optional[str] = None,
        cleared_at: Optional[str] = None,
    ) -> None:
        doc_dict = document.to_dict()
        inj_time = (
            injected_at
            or doc_dict.get("metadata", {}).get("injected_at")
            or datetime.now(timezone.utc).isoformat()
        )
        super().__init__(
            document_id=doc_dict["document_id"],
            title=doc_dict["title"],
            source=doc_dict["source"],
            content=doc_dict["content"],
            metadata=dict(doc_dict["metadata"]),
            status=status,
            injected_at=inj_time,
            cleared_at=cleared_at,
        )

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'InjectionRecord' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a standard dictionary representation."""
        return {
            "document_id": self["document_id"],
            "title": self["title"],
            "source": self["source"],
            "content": self["content"],
            "metadata": dict(self["metadata"]),
            "status": self["status"],
            "injected_at": self["injected_at"],
            "cleared_at": self["cleared_at"],
        }

    def to_document(self) -> Document:
        """Reconstruct a Document instance from this historical record."""
        return Document(
            document_id=self["document_id"],
            title=self["title"],
            source=self["source"],
            content=self["content"],
            metadata=dict(self["metadata"]),
        )


class RAGAgentConsole:
    """
    Interactive console session managing the RAG Agent, experimental documents,
    retriever state, and query history.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        top_k: int = DEFAULT_TOP_K,
        session_id: Optional[str] = None,
        model_adapter: Optional[ModelAdapter] = None,
        corpus_dir: Optional[Union[str, Path]] = None,
        export_dir: Optional[Union[str, Path]] = None,
        input_func: Optional[Callable[[str], str]] = None,
        output_func: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Initialize the interactive RAG console session.

        Args:
            model_name: Ollama model identifier.
            base_url: Localhost Ollama endpoint URL.
            timeout: Model generation timeout in seconds.
            top_k: Default number of chunks to retrieve.
            session_id: Optional correlation session identifier.
            model_adapter: Optional custom ModelAdapter (e.g. MockModel for tests).
            corpus_dir: Optional custom path to baseline corpus directory.
            export_dir: Optional directory for exported research reports.
            input_func: Function for reading user input (defaults to input).
            output_func: Function for writing console output (defaults to print).
        """
        self.session_id = session_id or f"rag-{uuid.uuid4().hex[:8]}"
        self.model_name = model_name
        self.base_url = base_url
        self.timeout = float(timeout)
        self.top_k = int(top_k)
        self.corpus_dir = Path(corpus_dir or get_default_corpus_dir()).resolve()
        self.export_dir = Path(export_dir or DEFAULT_EXPORT_DIR).resolve()

        self._input_func = input_func or input
        self._output_func = output_func or print

        self.session_start_time = datetime.now(timezone.utc)
        self.history: List[Dict[str, Any]] = []

        # Load baseline corpus (5 synthetic documents)
        self.baseline_documents: List[Document] = load_synthetic_corpus(self.corpus_dir)
        self.injected_documents: List[Document] = []
        self.injection_history: List[InjectionRecord] = []

        # Event collector for observability
        self.event_collector = EventCollector()
        self.prompt_builder = RAGPromptBuilder()

        # Initialize model adapter
        if model_adapter is not None:
            self.model: ModelAdapter = model_adapter
            self.model_name = getattr(model_adapter, "model_name", type(model_adapter).__name__)
        else:
            self.model = OllamaAdapter(
                model_name=self.model_name,
                base_url=self.base_url,
                timeout=self.timeout,
                options={"temperature": 0.0},
            )

        # Build initial retriever from baseline documents
        self.retriever = LocalRetriever.from_documents(self.baseline_documents, top_k=self.top_k)

        # Initialize RAGAgent
        self.agent = RAGAgent(
            model=self.model,
            retriever=self.retriever,
            event_collector=self.event_collector,
            prompt_builder=self.prompt_builder,
            system_prompt_version="rag-console-v1",
            top_k=self.top_k,
        )

    def _output(self, msg: str = "") -> None:
        """Helper to print output through configured output function."""
        self._output_func(msg)

    def _input(self, prompt: str = "") -> str:
        """Helper to read input through configured input function."""
        return self._input_func(prompt)

    def _rebuild_retriever(self) -> None:
        """
        Reconstruct the experimental retriever combining baseline documents
        and active in-memory injected documents.
        """
        combined = list(self.baseline_documents) + list(self.injected_documents)
        self.retriever = LocalRetriever.from_documents(combined, top_k=self.top_k)
        self.agent.retriever = self.retriever
        self.agent.top_k = self.top_k

    def get_startup_info(self) -> str:
        """Format the startup status block."""
        lines = [
            format_banner(),
            f"  Session ID:      {self.session_id}",
            f"  Model:           {self.model_name} ({self.base_url})",
            f"  Retriever:       LocalRetriever (BM25 ranking)",
            f"  Top-K:           {self.top_k}",
            f"  Corpus Location: {self.corpus_dir} ({len(self.baseline_documents)} baseline docs)",
            "",
            "Type a question directly or enter /help for available commands.",
            "Type /quit or /exit to leave the console.",
            "",
        ]
        return "\n".join(lines)

    def handle_query(self, query: str) -> RAGAgentResult:
        """
        Execute a user query through the RAG Agent, record history, and display report.

        Args:
            query: The user query text.

        Returns:
            The resulting RAGAgentResult.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            self._output("[Empty query. Please provide a question.]")
            return RAGAgentResult(user_query=query, error="Empty query.")

        result = self.agent.query(
            user_query=cleaned_query,
            session_id=self.session_id,
            top_k=self.top_k,
        )

        # Preserve query interaction in local session history
        history_entry = {
            "query_number": len(self.history) + 1,
            "user_query": cleaned_query,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": result.duration_seconds,
            "retrieved_document_ids": [c.document_id for c in result.retrieved_chunks],
            "retrieved_chunk_ids": [c.chunk_id for c in result.retrieved_chunks],
            "retrieval_scores": {c.chunk_id: c.score for c in result.retrieved_chunks},
            "retrieved_chunks": [c.to_dict() for c in result.retrieved_chunks],
            "prompt": result.prompt,
            "raw_model_response": result.model_response,
            "final_response": result.final_response,
            "error": result.error,
        }
        self.history.append(history_entry)

        # Display formatted report
        self._output(format_rag_result(result, self.top_k))
        return result

    def handle_command(self, command_line: str) -> bool:
        """
        Process a slash command.

        Args:
            command_line: The full command line input (e.g. '/help', '/ask ...').

        Returns:
            True to continue the REPL loop, False to exit cleanly.
        """
        raw = command_line.strip()
        if not raw:
            return True

        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/quit", "/exit"):
            self._output("\n[Exiting KAS RAG Console. Goodbye!]\n")
            return False

        elif cmd == "/help":
            self._output(HELP_TEXT)
            return True

        elif cmd == "/ask":
            if not args:
                self._output("[Usage: /ask <your question>]")
            else:
                self.handle_query(args)
            return True

        elif cmd == "/docs":
            self._display_docs()
            return True

        elif cmd == "/inject":
            self._interactive_inject()
            return True

        elif cmd == "/injections":
            self._display_injections()
            return True

        elif cmd == "/clear-injections":
            self._clear_injections()
            return True

        elif cmd == "/model":
            self._handle_model_command(args)
            return True

        elif cmd == "/topk":
            self._handle_topk_command(args)
            return True

        elif cmd == "/events":
            self._display_events()
            return True

        elif cmd == "/export":
            prefix = args if args else "rag_session"
            self.export_session(prefix=prefix)
            return True

        else:
            self._output(f"\n[Unknown command '{command_line}'. Type /help for available commands.]\n")
            return True

    def _display_docs(self) -> None:
        """Display currently loaded baseline corpus documents."""
        self._output("\n─── BASELINE SYNTHETIC CORPUS ───")
        self._output(f"Directory: {self.corpus_dir}")
        self._output(f"Total baseline documents: {len(self.baseline_documents)}\n")

        for idx, doc in enumerate(self.baseline_documents, 1):
            chunks = [c for c in self.retriever.chunks if c.document_id == doc.document_id]
            classification = doc.metadata.get("classification", "N/A")
            dept = doc.metadata.get("department", "N/A")
            version = doc.metadata.get("version", "N/A")

            self._output(f"[{idx}] Document ID:   {doc.document_id}")
            self._output(f"    Title:         {doc.title}")
            self._output(f"    Classification:{classification}")
            self._output(f"    Department:    {dept} (v{version})")
            self._output(f"    Chunks:        {len(chunks)}")
            self._output(f"    Source:        {doc.source}")
            self._output("")
        self._output("─────────────────────────────────\n")

    def inject_document(self, document: Document) -> InjectionRecord:
        """
        Inject an ephemeral document into the active retrieval collection and
        record its complete payload and metadata in the session historical collection.

        Args:
            document: Document instance to inject.

        Returns:
            The created InjectionRecord.
        """
        self.injected_documents.append(document)
        record = InjectionRecord(
            document=document,
            status="active",
            injected_at=document.metadata.get("injected_at") or datetime.now(timezone.utc).isoformat(),
            cleared_at=None,
        )
        self.injection_history.append(record)
        self._rebuild_retriever()
        return record

    def _interactive_inject(self) -> None:
        """Prompt user interactively to inject an ephemeral experimental document."""
        self._output("\n─── INJECT EXPERIMENTAL DOCUMENT ───")

        existing_ids = {d.document_id for d in self.baseline_documents} | {
            d.document_id for d in self.injected_documents
        }

        # 1. Document ID collection with validation and field restart
        doc_id = ""
        while True:
            try:
                raw_id = self._input("Document ID: ")
            except (EOFError, KeyboardInterrupt):
                self._output("\n[Injection cancelled.]\n")
                return

            if not raw_id.strip():
                self._output("[Injection cancelled: Document ID cannot be empty.]\n")
                return

            is_valid, error_msg, is_dup = validate_document_id(raw_id, existing_ids)
            if not is_valid:
                if is_dup:
                    self._output(f"[Injection cancelled: {error_msg}]\n")
                    return
                else:
                    self._output(f"[Invalid Document ID: {error_msg} Please try again.]\n")
                    continue

            doc_id = raw_id.strip()
            break

        # 2. Document Title collection with validation and field restart
        title = ""
        while True:
            try:
                raw_title = self._input("Document Title: ")
            except (EOFError, KeyboardInterrupt):
                self._output("\n[Injection cancelled.]\n")
                return

            is_valid, error_msg = validate_document_title(raw_title)
            if not is_valid:
                self._output(f"[Invalid Document Title: {error_msg} Please try again.]\n")
                continue

            stripped_title = raw_title.strip()
            title = stripped_title if stripped_title else f"Experimental Injected Document ({doc_id})"
            break

        # 3. Multiline Document Content collection
        self._output("Enter document content (type '.end' on a new line to finish):")
        content_lines: List[str] = []
        terminated = False

        while True:
            try:
                line = self._input()
            except (EOFError, KeyboardInterrupt):
                self._output(
                    "\n[Injection cancelled: Interrupted or reached end-of-input without '.end' terminator.]\n"
                )
                return

            # Support single line or embedded multiline chunk
            sublines = line.splitlines() if ("\n" in line or "\r" in line) else [line]
            for subline in sublines:
                if subline.strip() == ".end":
                    terminated = True
                    break
                # Do not execute slash commands while collecting content; preserve as literal content
                content_lines.append(subline)

            if terminated:
                break

        if not terminated:
            self._output(
                "[Injection cancelled: Document content collection was not terminated with '.end'.]\n"
            )
            return

        content = "\n".join(content_lines).strip()
        if not content:
            self._output("[Injection cancelled: Document content cannot be empty.]\n")
            return

        # Build experimental document in-memory only
        metadata = {
            "category": "experimental",
            "department": "security_research",
            "classification": "synthetic_injected",
            "is_ephemeral": True,
            "session_id": self.session_id,
            "injected_at": datetime.now(timezone.utc).isoformat(),
            "origin": "interactive_console_injection",
        }
        doc = Document(
            document_id=doc_id,
            title=title,
            source=f"in_memory_ephemeral://{self.session_id}/{doc_id}",
            content=content,
            metadata=metadata,
        )

        self.inject_document(doc)

        self._output(
            f"[Document '{doc_id}' successfully injected into experimental retriever. "
            f"Total indexed documents: {len(self.retriever.documents)}]\n"
        )

    def _display_injections(self) -> None:
        """List experimental documents currently injected into the session."""
        self._output("\n─── EXPERIMENTAL INJECTED DOCUMENTS ───")
        if not self.injected_documents:
            self._output("[No experimental documents currently injected.]")
            if self.injection_history:
                cleared_count = len([r for r in self.injection_history if r.get("status") == "cleared"])
                self._output(
                    f"Note: {len(self.injection_history)} historical document(s) preserved in session log "
                    f"({cleared_count} cleared). Use /export to review full evidence."
                )
            self._output("Use /inject to introduce an experimental synthetic document.\n")
            return

        self._output(f"Total experimental documents: {len(self.injected_documents)}\n")
        for idx, doc in enumerate(self.injected_documents, 1):
            chunks = [c for c in self.retriever.chunks if c.document_id == doc.document_id]
            preview = doc.content[:100].replace("\n", " ")
            if len(doc.content) > 100:
                preview += "..."

            self._output(f"[{idx}] Document ID:   {doc.document_id} [EXPERIMENTAL INJECTION]")
            self._output(f"    Title:         {doc.title}")
            self._output(f"    Classification:{doc.metadata.get('classification', 'synthetic_injected')}")
            self._output(f"    Injected At:   {doc.metadata.get('injected_at', 'N/A')}")
            self._output(f"    Chunks:        {len(chunks)}")
            self._output(f"    Preview:       {preview}")
            self._output("")
        self._output("───────────────────────────────────────\n")

    def clear_injections(self) -> int:
        """
        Clear all active injected documents and restore retriever to baseline documents
        while preserving all historical injection evidence and metadata.

        Returns:
            Number of active documents cleared.
        """
        count = len(self.injected_documents)
        cleanup_time = datetime.now(timezone.utc).isoformat()

        for record in self.injection_history:
            if record.get("status") == "active":
                record["status"] = "cleared"
                record["cleared_at"] = cleanup_time
                if "metadata" in record and isinstance(record["metadata"], dict):
                    record["metadata"]["cleared_at"] = cleanup_time
                    record["metadata"]["is_active"] = False

        self.injected_documents.clear()
        self._rebuild_retriever()
        return count

    def _clear_injections(self) -> None:
        """Clear all active injected documents and restore retriever to baseline documents."""
        count = self.clear_injections()
        self._output(
            f"\n[Cleared {count} experimental injected document(s). "
            f"Retriever restored to {len(self.retriever.documents)} baseline documents.]\n"
        )

    def _handle_model_command(self, model_arg: str) -> None:
        """Display or switch the active Ollama model."""
        if not model_arg:
            self._output(
                f"\nCurrent model: {self.model_name} "
                f"(Endpoint: {self.base_url}, Timeout: {self.timeout}s)\n"
            )
            return

        try:
            validated_name = _validate_model_name(model_arg)
        except OllamaConfigurationError as e:
            self._output(f"\n[Invalid model name '{model_arg}': {e}]\n")
            return

        try:
            new_adapter = OllamaAdapter(
                model_name=validated_name,
                base_url=self.base_url,
                timeout=self.timeout,
                options={"temperature": 0.0},
            )
            self.model = new_adapter
            self.model_name = validated_name
            self.agent.model = new_adapter
            self._output(f"\n[Active model switched to '{validated_name}'.]\n")
        except Exception as e:
            self._output(f"\n[Failed to switch model to '{validated_name}': {e}]\n")

    def _handle_topk_command(self, topk_arg: str) -> None:
        """Display or update the retrieval top-K value."""
        if not topk_arg:
            self._output(f"\nCurrent retrieval Top-K: {self.top_k}\n")
            return

        try:
            k = int(topk_arg)
            if k <= 0:
                raise ValueError("Top-K must be a positive integer.")
            self.top_k = k
            self.retriever.top_k = k
            self.agent.top_k = k
            self._output(f"\n[Retrieval Top-K updated to {k}.]\n")
        except ValueError as e:
            self._output(f"\n[Invalid Top-K value '{topk_arg}': must be an integer > 0. {e}]\n")

    def _display_events(self) -> None:
        """Display recorded observability events for this session."""
        events = self.agent.get_logs(session_id=self.session_id)
        self._output(f"\n─── SESSION OBSERVABILITY EVENTS ({self.session_id}) ───")
        if not events:
            self._output("[No observability events recorded for this session yet.]\n")
            return

        for idx, ev in enumerate(events, 1):
            info_parts = [f"Type: {ev.event_type}", f"State: {ev.agent_state}"]
            if ev.duration_seconds is not None:
                info_parts.append(f"Duration: {ev.duration_seconds:.3f}s")
            if ev.retrieved_chunks is not None:
                info_parts.append(f"Chunks: {len(ev.retrieved_chunks)}")
            if ev.error:
                info_parts.append(f"Error: {ev.error}")

            details = " | ".join(info_parts)
            self._output(f"[{idx}] {ev.timestamp} | {details}")
        self._output("────────────────────────────────────────────────────────\n")

    def export_session(self, prefix: str = "rag_session") -> Tuple[Path, Path]:
        """
        Export session history and metadata to structured JSON and Markdown files.

        Args:
            prefix: Filename prefix for exported reports.

        Returns:
            Tuple of (json_path, markdown_path).
        """
        self.export_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc)
        timestamp_slug = now.strftime("%Y%m%d_%H%M%S_%f")[:19]
        total_duration = round((now - self.session_start_time).total_seconds(), 3)

        json_path = self.export_dir / f"{prefix}_{self.session_id}_{timestamp_slug}.json"
        md_path = self.export_dir / f"{prefix}_{self.session_id}_{timestamp_slug}.md"

        events = [e.to_dict() for e in self.agent.get_logs(session_id=self.session_id)]

        # Prepare injection collections
        active_injected = [doc.to_dict() for doc in self.injected_documents]
        historical_injections = [
            r.to_dict() if hasattr(r, "to_dict") else dict(r)
            for r in self.injection_history
        ]
        cleared_injections = [
            r.to_dict() if hasattr(r, "to_dict") else dict(r)
            for r in self.injection_history
            if r.get("status") == "cleared"
        ]

        # 1. Build JSON structure
        session_data = {
            "session_id": self.session_id,
            "exported_at": now.isoformat(),
            "session_start_time": self.session_start_time.isoformat(),
            "total_duration_seconds": total_duration,
            "model_configuration": {
                "model_name": self.model_name,
                "base_url": self.base_url,
                "timeout": self.timeout,
            },
            "top_k": self.top_k,
            "corpus_summary": {
                "baseline_corpus_dir": str(self.corpus_dir),
                "baseline_documents_count": len(self.baseline_documents),
                "active_injected_documents_count": len(self.injected_documents),
                "injected_documents_count": len(self.injected_documents),
                "total_injected_documents_count": len(self.injection_history),
                "cleared_injected_documents_count": len(cleared_injections),
                "total_indexed_chunks": len(self.retriever.chunks),
            },
            "active_injected_documents": active_injected,
            "injected_documents": active_injected,
            "all_injected_documents": historical_injections,
            "injection_history": historical_injections,
            "cleared_injected_documents": cleared_injections,
            "query_count": len(self.history),
            "queries": self.history,
            "observability_events": events,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, default=str)

        # 2. Build Markdown report
        md_lines = [
            f"# KAS RAG Agent — Research Session Report",
            "",
            f"**Session ID:** `{self.session_id}`  ",
            f"**Exported At:** `{now.isoformat()}`  ",
            f"**Duration:** `{total_duration}s`  ",
            f"**Model:** `{self.model_name}` (`{self.base_url}`)  ",
            f"**Top-K:** `{self.top_k}`  ",
            f"**Baseline Documents:** {len(self.baseline_documents)}  ",
            f"**Active Injected Documents:** {len(self.injected_documents)}  ",
            f"**Injected Experimental Documents:** {len(self.injected_documents)}  ",
            f"**Total Injected Documents (Session):** {len(self.injection_history)}  ",
            f"**Cleared Injected Documents:** {len(cleared_injections)}  ",
            "",
            "---",
            "",
            "## 1. Experimental Injected Documents",
            "",
        ]

        if not self.injection_history:
            md_lines.append("*No experimental documents were injected during this session.*")
        else:
            md_lines.extend([
                f"### 1.1 All Injected Documents ({len(self.injection_history)} total)",
                "",
                "| Document ID | Title | Status | Injected At | Cleared At |",
                "|---|---|---|---|---|",
            ])
            for rec in self.injection_history:
                did = rec.get("document_id", "N/A")
                title = rec.get("title", "N/A")
                status = rec.get("status", "unknown").upper()
                inj_time = rec.get("injected_at", "N/A")
                clr_time = rec.get("cleared_at") or "N/A"
                md_lines.append(f"| `{did}` | {title} | {status} | {inj_time} | {clr_time} |")

            md_lines.extend([
                "",
                f"### 1.2 Active Injected Documents ({len(self.injected_documents)})",
                "",
            ])
            if not self.injected_documents:
                md_lines.append("*No experimental documents are currently active in the retriever.*")
            else:
                md_lines.extend([
                    "| Document ID | Title | Injected At |",
                    "|---|---|---|",
                ])
                for doc in self.injected_documents:
                    inj_time = doc.metadata.get("injected_at", "N/A")
                    md_lines.append(f"| `{doc.document_id}` | {doc.title} | {inj_time} |")

            md_lines.extend([
                "",
                f"### 1.3 Cleared Injected Documents ({len(cleared_injections)})",
                "",
            ])
            if not cleared_injections:
                md_lines.append("*No injected documents were cleared during this session.*")
            else:
                md_lines.extend([
                    "| Document ID | Title | Injected At | Cleared At |",
                    "|---|---|---|---|",
                ])
                for rec in cleared_injections:
                    did = rec.get("document_id", "N/A")
                    title = rec.get("title", "N/A")
                    inj_time = rec.get("injected_at", "N/A")
                    clr_time = rec.get("cleared_at", "N/A")
                    md_lines.append(f"| `{did}` | {title} | {inj_time} | {clr_time} |")

            md_lines.extend([
                "",
                "### 1.4 Historical Injected Document Payloads & Evidence",
                "",
            ])
            for rec in self.injection_history:
                did = rec.get("document_id", "N/A")
                title = rec.get("title", "N/A")
                status = rec.get("status", "unknown").upper()
                inj_time = rec.get("injected_at", "N/A")
                clr_time = rec.get("cleared_at") or "N/A"
                source = rec.get("source", "N/A")
                content = rec.get("content", "")
                md = rec.get("metadata", {})

                md_lines.append(f"#### Document: `{did}` ({status})")
                md_lines.append(f"- **Document ID:** `{did}`")
                md_lines.append(f"- **Title:** {title}")
                md_lines.append(f"- **Status:** `{status}`")
                md_lines.append(f"- **Injected At:** {inj_time}")
                md_lines.append(f"- **Cleared At:** {clr_time}")
                md_lines.append(f"- **Source:** `{source}`")
                md_lines.append(f"- **Classification:** `{md.get('classification', 'N/A')}`")
                md_lines.append(f"- **Ephemeral:** `{md.get('is_ephemeral', False)}`")
                md_lines.append("- **Injected Payload / Content:**")
                md_lines.append("  ```text")
                for cline in content.splitlines():
                    md_lines.append(f"  {cline}")
                md_lines.append("  ```")
                md_lines.append("")

        md_lines.extend([
            "---",
            "",
            f"## 2. Interaction History ({len(self.history)} queries)",
            "",
        ])

        if not self.history:
            md_lines.append("*No queries executed during this session.*")
        else:
            for item in self.history:
                q_num = item.get("query_number", 1)
                q_text = item.get("user_query", "")
                dur = item.get("duration_seconds", 0.0)
                err = item.get("error")
                resp = item.get("final_response") or "[None]"

                md_lines.append(f"### Query {q_num}: \"{q_text}\"")
                md_lines.append(f"- **Duration:** {dur:.3f}s" if dur else "- **Duration:** N/A")
                if err:
                    md_lines.append(f"- **Error:** `{err}`")

                md_lines.append("- **Retrieved Chunks:**")
                chunk_ids = item.get("retrieved_chunk_ids", [])
                doc_ids = item.get("retrieved_document_ids", [])
                scores = item.get("retrieval_scores", {})

                if not chunk_ids:
                    md_lines.append("  - *No chunks retrieved*")
                else:
                    md_lines.append("  | Rank | Document ID | Chunk ID | BM25 Score | Origin |")
                    md_lines.append("  |---|---|---|---|---|")
                    for rank, cid in enumerate(chunk_ids, 1):
                        did = doc_ids[rank - 1] if rank - 1 < len(doc_ids) else "N/A"
                        score = scores.get(cid, 0.0)
                        is_active_inj = any(d.document_id == did for d in self.injected_documents)
                        is_hist_inj = any(
                            (d.document_id if hasattr(d, "document_id") else d["document_id"]) == did
                            for d in self.injection_history
                        )
                        if is_active_inj:
                            tag = "**EXPERIMENTAL INJECTION**"
                        elif is_hist_inj:
                            tag = "**EXPERIMENTAL INJECTION (CLEARED)**"
                        else:
                            tag = "BASELINE"
                        md_lines.append(f"  | {rank} | `{did}` | `{cid}` | {score:.6f} | {tag} |")

                md_lines.extend([
                    "",
                    "- **Final Response:**",
                    "  ```text",
                    f"  {resp.strip()}",
                    "  ```",
                    "",
                ])

        md_lines.extend([
            "---",
            "",
            f"## 3. Observability Event Log ({len(events)} events)",
            "",
        ])

        for idx, ev in enumerate(events, 1):
            etype = ev.get("event_type", "unknown")
            state = ev.get("agent_state", "unknown")
            ts = ev.get("timestamp", "")
            md_lines.append(f"{idx}. `{ts}` | `{etype}` (state: `{state}`)")

        md_lines.append("")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        self._output(f"\n[Session artifacts successfully exported:]")
        self._output(f"  JSON:     {json_path}")
        self._output(f"  Markdown: {md_path}\n")
        return json_path, md_path

    def run_repl(self) -> None:
        """Run the interactive command-line REPL loop."""
        self._output(self.get_startup_info())

        while True:
            try:
                user_input = self._input("RAG > ").strip()
            except EOFError:
                self._output("\n[Exiting KAS RAG Console. Goodbye!]\n")
                break
            except KeyboardInterrupt:
                self._output("\n[Session interrupted. Type /quit or press Ctrl+D to exit.]\n")
                continue

            if not user_input:
                continue

            if user_input.startswith("/"):
                should_continue = self.handle_command(user_input)
                if not should_continue:
                    break
            else:
                self.handle_query(user_input)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI startup arguments."""
    parser = argparse.ArgumentParser(
        description="KAS RAG Agent — Interactive Security Console",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Localhost Ollama model identifier (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_BASE_URL,
        help=f"Localhost Ollama endpoint URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Retrieval top-K chunk count (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session identifier (default: auto-generated)",
    )
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="Optional custom corpus directory path (default: synthetic_data/rag/documents)",
    )
    return parser.parse_args(args)


def main() -> None:
    """CLI entry point for launching the KAS RAG Agent console."""
    args = parse_args()

    console = RAGAgentConsole(
        model_name=args.model,
        base_url=args.endpoint,
        timeout=args.timeout,
        top_k=args.top_k,
        session_id=args.session_id,
        corpus_dir=args.corpus_dir,
    )

    console.run_repl()


if __name__ == "__main__":
    main()
