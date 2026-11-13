import time
import uuid
from typing import Any, Dict, List, Optional

from agents.rag_agent.result import RAGAgentResult
from core.logging.collector import EventCollector
from core.logging.events import Event
from core.models.base import ModelAdapter
from core.rag.prompt import RAGPromptBuilder
from core.rag.retriever import LocalRetriever
from core.rag.validator import RAGSecurityValidator


class RAGAgent:
    """
    Controlled Retrieval-Augmented Generation (RAG) agent baseline.

    Coordinates between a deterministic LocalRetriever, a RAGPromptBuilder,
    and a ModelAdapter to answer user queries grounded on a controlled corpus.

    Security Semantics:
    Retrieved documents are treated strictly as UNTRUSTED DATA and are demarcated
    in the prompt's context block, completely separated from trusted system instructions.
    Response-level validation enforces the RAG-01 security rule against indirect injection.
    """

    def __init__(
        self,
        model: ModelAdapter,
        retriever: LocalRetriever,
        event_collector: Optional[EventCollector] = None,
        prompt_builder: Optional[RAGPromptBuilder] = None,
        system_prompt_version: Optional[str] = "rag-v1",
        top_k: Optional[int] = None,
        validator: Optional[RAGSecurityValidator] = None,
        enable_response_validation: bool = True,
    ) -> None:
        """
        Initialize the RAGAgent.

        Args:
            model: ModelAdapter instance to generate responses.
            retriever: LocalRetriever configured with a controlled corpus.
            event_collector: Optional EventCollector for observability.
            prompt_builder: Optional RAGPromptBuilder for structuring prompts.
            system_prompt_version: Version identifier for prompt tracking.
            top_k: Optional default top_k override for retrieval.
            validator: Optional RAGSecurityValidator for response-level validation.
            enable_response_validation: Whether to run response-level validation (default True).
        """
        self.model = model
        self.retriever = retriever
        self.event_collector = event_collector
        self.prompt_builder = prompt_builder or RAGPromptBuilder()
        self.system_prompt_version = system_prompt_version
        self.top_k = top_k
        self.validator = validator or RAGSecurityValidator()
        self.enable_response_validation = enable_response_validation
        self._state: str = "idle"
        self._is_running: bool = True

    def start(self) -> "RAGAgent":
        """Start the agent and set lifecycle state to running."""
        self._is_running = True
        self._state = "running"
        return self

    def stop(self) -> "RAGAgent":
        """Stop the agent and set lifecycle state to stopped."""
        self._is_running = False
        self._state = "stopped"
        return self

    def reset(self) -> None:
        """
        Reset agent lifecycle state and clear recorded event logs.
        """
        self._state = "idle"
        if self.event_collector is not None:
            self.event_collector.clear()

    def get_state(self) -> Dict[str, Any]:
        """Return the current agent lifecycle and configuration state."""
        return {
            "state": self._state,
            "is_running": self._is_running,
            "model": type(self.model).__name__,
            "retriever": type(self.retriever).__name__,
            "chunk_count": len(self.retriever.chunks),
            "document_count": len(self.retriever.documents),
            "top_k": self.top_k or self.retriever.top_k,
            "has_event_collector": self.event_collector is not None,
            "system_prompt_version": self.system_prompt_version,
            "enable_response_validation": self.enable_response_validation,
        }

    def get_logs(self, session_id: Optional[str] = None) -> List[Event]:
        """Retrieve recorded observability events from the event collector."""
        if self.event_collector is not None:
            return self.event_collector.get_events(session_id=session_id)
        return []

    def clear_logs(self) -> None:
        """Clear recorded observability events from the event collector."""
        if self.event_collector is not None:
            self.event_collector.clear()

    def _emit_event(self, session_id: str, **kwargs: Any) -> None:
        """Record an event if an event collector is configured."""
        if self.event_collector is not None:
            event = Event(
                session_id=session_id,
                system_prompt_version=self.system_prompt_version,
                model=type(self.model).__name__,
                **kwargs,
            )
            self.event_collector.record(event)

    def query(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> RAGAgentResult:
        """
        Execute the RAG pipeline for a user query:
        1. Retrieve relevant chunks from the controlled corpus
        2. Construct a prompt cleanly distinguishing untrusted context from trusted instructions
        3. Invoke the ModelAdapter
        4. Return a structured RAGAgentResult

        Args:
            user_query: The query string from the user.
            session_id: Optional session identifier for correlating events.
            top_k: Optional override for the number of chunks to retrieve.

        Returns:
            A RAGAgentResult containing the query, retrieved chunks, model response,
            final response, and any error message.
        """
        sid = session_id or uuid.uuid4().hex[:12]
        start_time = time.monotonic()

        # Lifecycle check
        if not self._is_running:
            error_msg = "Agent is stopped. Call start() before querying."
            self._emit_event(
                session_id=sid,
                event_type="error",
                error=error_msg,
                agent_state="stopped",
            )
            return RAGAgentResult(
                user_query=str(user_query),
                error=error_msg,
                session_id=sid,
                duration_seconds=round(time.monotonic() - start_time, 4),
            )

        # Input validation
        if not isinstance(user_query, str):
            error_msg = f"User query must be a string, got {type(user_query).__name__}."
            self._emit_event(
                session_id=sid,
                event_type="error",
                error=error_msg,
                agent_state="error",
            )
            return RAGAgentResult(
                user_query=str(user_query),
                error=error_msg,
                session_id=sid,
                duration_seconds=round(time.monotonic() - start_time, 4),
            )

        # Emit initial query event
        self._emit_event(
            session_id=sid,
            event_type="rag_query",
            user_input=user_query,
            agent_state="received_query",
        )

        retrieval_query = user_query
        effective_top_k = top_k if top_k is not None else self.top_k

        # 1. Retrieve chunks
        try:
            retrieved_chunks = self.retriever.retrieve(
                query=retrieval_query,
                top_k=effective_top_k,
            )
        except Exception as e:
            error_msg = f"Retrieval failed: {e}"
            self._emit_event(
                session_id=sid,
                event_type="error",
                error=error_msg,
                agent_state="error",
            )
            return RAGAgentResult(
                user_query=user_query,
                error=error_msg,
                session_id=sid,
                retrieval_query=retrieval_query,
                duration_seconds=round(time.monotonic() - start_time, 4),
            )

        # Record retrieval event
        self._emit_event(
            session_id=sid,
            event_type="retrieval",
            user_input=user_query,
            retrieval_query=retrieval_query,
            retrieved_chunks=[c.to_dict() for c in retrieved_chunks],
            retrieval_scores={c.chunk_id: c.score for c in retrieved_chunks},
            agent_state="retrieval_completed",
        )

        # 2. Construct prompt
        prompt = self.prompt_builder.build_prompt(
            user_query=user_query,
            retrieved_chunks=retrieved_chunks,
        )

        # 3. Model generation
        try:
            raw_response = self.model.generate(prompt)
        except Exception as e:
            error_msg = f"Model generation failed: {e}"
            self._emit_event(
                session_id=sid,
                event_type="error",
                error=error_msg,
                agent_state="error",
            )
            return RAGAgentResult(
                user_query=user_query,
                retrieved_chunks=retrieved_chunks,
                prompt=prompt,
                error=error_msg,
                session_id=sid,
                retrieval_query=retrieval_query,
                duration_seconds=round(time.monotonic() - start_time, 4),
            )

        duration = round(time.monotonic() - start_time, 4)

        # 4. Response-level validation and layered defense
        validation_result = None
        if self.enable_response_validation and self.validator is not None:
            validation_result = self.validator.validate(
                query=user_query,
                retrieved_chunks=retrieved_chunks,
                raw_response=raw_response,
            )

        if validation_result and validation_result.is_compromised:
            final_response = validation_result.sanitized_response
            self._emit_event(
                session_id=sid,
                event_type="security_alert",
                security_event="indirect_injection_mitigated",
                user_input=user_query,
                model_response=raw_response,
                final_response=final_response,
                agent_state="threat_mitigated",
            )
        else:
            final_response = raw_response

        # 5. Final response and completion event
        self._emit_event(
            session_id=sid,
            event_type="rag_result",
            user_input=user_query,
            model_response=raw_response,
            final_response=final_response,
            duration_seconds=duration,
            agent_state="completed",
        )

        return RAGAgentResult(
            user_query=user_query,
            retrieved_chunks=retrieved_chunks,
            model_response=raw_response,
            final_response=final_response,
            error=None,
            session_id=sid,
            retrieval_query=retrieval_query,
            prompt=prompt,
            duration_seconds=duration,
        )

    def send_message(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> RAGAgentResult:
        """
        Alias for query() for interface compatibility with ToolUsingAgent.

        Args:
            user_input: The user prompt / question.
            session_id: Optional session identifier.
            top_k: Optional top_k override.

        Returns:
            A RAGAgentResult instance.
        """
        return self.query(
            user_query=user_input,
            session_id=session_id,
            top_k=top_k,
        )
