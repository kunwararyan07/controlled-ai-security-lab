from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from core.models.tool_call import ToolCall


@dataclass
class AgentResult:
    """
    Structured result of a ToolUsingAgent execution.

    Attributes:
        session_id: The session or request identifier for correlating events.
        final_response: The final response content (if normal response or after tool execution).
        tool_call: The structured tool call requested by the model, if any.
        authorization_allowed: Whether authorization explicitly permitted the tool call.
        authorization_denied: Whether authorization explicitly denied the tool call.
        authorization_reason: Explanation from policy if authorization was evaluated.
        tool_executed: Whether the tool was actually executed.
        tool_result: The output returned by the executed tool, if executed.
        error: Error message if an error occurred during execution.
    """

    session_id: Optional[str] = None
    final_response: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    authorization_allowed: bool = False
    authorization_denied: bool = False
    authorization_reason: Optional[str] = None
    tool_executed: bool = False
    tool_result: Optional[Any] = None
    error: Optional[str] = None
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.tool_call is not None and not self.tool_calls:
            self.tool_calls = [self.tool_call]
        if self.tool_result is not None and not self.tool_results:
            self.tool_results = [self.tool_result]

    @property
    def tool_requested(self) -> bool:
        """Whether a tool was requested by the model."""
        return self.tool_call is not None or len(self.tool_calls) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent result to a dictionary representation."""
        return {
            "session_id": self.session_id,
            "final_response": self.final_response,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_requested": self.tool_requested,
            "authorization_allowed": self.authorization_allowed,
            "authorization_denied": self.authorization_denied,
            "authorization_reason": self.authorization_reason,
            "tool_executed": self.tool_executed,
            "tool_result": self.tool_result,
            "tool_results": self.tool_results,
            "error": self.error,
        }
