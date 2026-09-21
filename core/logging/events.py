from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _get_utc_now_iso() -> str:
    """Return the current UTC timestamp formatted as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Event:
    """
    Structured event representation for agent observability.

    All fields other than timestamp are optional to support various stages
    of agent interaction without requiring rigid schemas.
    """

    timestamp: str = field(default_factory=_get_utc_now_iso)
    session_id: Optional[str] = None
    event_type: Optional[str] = None
    user_input: Optional[str] = None
    model: Optional[str] = None
    system_prompt_version: Optional[str] = None
    agent_state: Optional[str] = None
    tool_call: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    authorization_decision: Optional[Dict[str, Any]] = None
    execution_result: Optional[Any] = None
    final_response: Optional[str] = None
    error: Optional[str] = None
    security_event: Optional[str] = None
    duration_seconds: Optional[float] = None

    def to_dict(self, include_none: bool = True) -> Dict[str, Any]:
        """
        Convert event to a dictionary representation.

        Args:
            include_none: If False, fields with None values will be omitted.
        """
        data = asdict(self)
        if not include_none:
            return {k: v for k, v in data.items() if v is not None}
        return data
