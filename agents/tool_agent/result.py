from dataclasses import dataclass
from typing import Any, Dict, Optional
from core.models.tool_call import ToolCall


@dataclass
class AgentResult:
    """
    Structured result of a ToolUsingAgent execution.

    Attributes:
        final_response: The final response content (if normal response or after tool execution).
        tool_call: The structured tool call requested by the model, if any.
        authorization_denied: Whether authorization denied the tool call.
        authorization_reason: Explanation from policy if authorization was denied.
        tool_executed: Whether the tool was actually executed.
        tool_result: The output returned by the executed tool, if executed.
        error: Error message if an error occurred during execution.
    """

    final_response: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    authorization_denied: bool = False
    authorization_reason: Optional[str] = None
    tool_executed: bool = False
    tool_result: Optional[Any] = None
    error: Optional[str] = None

    @property
    def tool_requested(self) -> bool:
        """Whether a tool was requested by the model."""
        return self.tool_call is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent result to a dictionary representation."""
        return {
            "final_response": self.final_response,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "tool_requested": self.tool_requested,
            "authorization_denied": self.authorization_denied,
            "authorization_reason": self.authorization_reason,
            "tool_executed": self.tool_executed,
            "tool_result": self.tool_result,
            "error": self.error,
        }
