from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass(frozen=True)
class ToolCall:
    """
    Structured representation of a model-generated tool call request.

    Attributes:
        tool_name: The name of the tool requested.
        arguments: Structured dictionary of input arguments for the tool.
    """

    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool call to dictionary format."""
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
        }
