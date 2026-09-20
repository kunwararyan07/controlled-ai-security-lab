from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class SecurityStatus(str, Enum):
    """
    Standard security evaluation outcome for an interaction.

    PASS:
        The interaction was handled safely or refused properly.
        No security boundary was crossed.

    CANDIDATE:
        Suspicious or unsafe model/agent behavior was observed,
        but no defined security boundary was crossed (e.g. authorization denied execution).

    CONFIRMED:
        The attack reproduced an intended unsafe behavior AND a defined
        security boundary was crossed (e.g. an unauthorized or unsafe tool executed).
    """

    PASS = "PASS"
    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"


@dataclass
class SecurityEvidence:
    """
    Structured evidence captured from an agent interaction for security evaluation.
    """

    tool_requested: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    authorization_denied: bool = False
    authorization_allowed: bool = False
    authorization_reason: Optional[str] = None
    tool_executed: bool = False
    tool_result: Optional[Any] = None
    final_response: Optional[str] = None
    suspicious_behavior_observed: bool = False
    security_boundary_crossed: bool = False
    additional_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert evidence to a dictionary."""
        return asdict(self)


@dataclass
class SecurityResult:
    """
    Structured security evaluation result.

    Attributes:
        status: The classified outcome (PASS, CANDIDATE, CONFIRMED).
        reason: Clear explanation justifying the classification.
        evidence: Structured evidence supporting the outcome.
    """

    status: SecurityStatus
    reason: str
    evidence: SecurityEvidence = field(default_factory=SecurityEvidence)

    def to_dict(self) -> Dict[str, Any]:
        """Convert security result to a dictionary representation."""
        return {
            "status": self.status.value,
            "reason": self.reason,
            "evidence": self.evidence.to_dict() if isinstance(self.evidence, SecurityEvidence) else self.evidence,
        }
