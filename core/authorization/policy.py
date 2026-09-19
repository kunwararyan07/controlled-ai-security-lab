from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Set


@dataclass(frozen=True)
class AuthorizationDecision:
    """
    Structured result of an authorization evaluation.

    Attributes:
        allowed: Whether the requested action is permitted.
        reason: Explanation of why the action was permitted or denied.
    """

    allowed: bool
    reason: str

    def __bool__(self) -> bool:
        return self.allowed

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to a dictionary representation."""
        return {
            "allowed": self.allowed,
            "reason": self.reason,
        }


class AuthorizationPolicy(ABC):
    """
    Abstract base interface for tool authorization policies.

    Enables provider-independent authorization decisions before tool execution.
    """

    @abstractmethod
    def evaluate(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthorizationDecision:
        """
        Evaluate whether a tool execution request is authorized.

        Args:
            tool_name: The name of the tool requested.
            arguments: Structured arguments provided for tool execution.
            context: Optional contextual information (user, session, environment).

        Returns:
            An AuthorizationDecision indicating whether the request is allowed and why.
        """
        raise NotImplementedError


class AllowlistAuthorizationPolicy(AuthorizationPolicy):
    """
    Deterministic authorization policy that permits tools based on an explicit allowlist.

    Any tool not in the allowlist is explicitly denied.
    """

    def __init__(self, allowed_tools: Optional[Iterable[str]] = None) -> None:
        self.allowed_tools: Set[str] = set(allowed_tools) if allowed_tools is not None else set()

    def add_tool(self, tool_name: str) -> None:
        """Add a tool name to the allowlist."""
        self.allowed_tools.add(tool_name)

    def remove_tool(self, tool_name: str) -> None:
        """Remove a tool name from the allowlist."""
        self.allowed_tools.discard(tool_name)

    def evaluate(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthorizationDecision:
        """
        Evaluate tool execution against the allowlist.

        Args:
            tool_name: The name of the tool requested.
            arguments: Structured arguments provided for tool execution.
            context: Optional contextual information.

        Returns:
            AuthorizationDecision with allowed status and reason.
        """
        if tool_name in self.allowed_tools:
            return AuthorizationDecision(
                allowed=True,
                reason=f"Tool '{tool_name}' is permitted by allowlist policy.",
            )
        return AuthorizationDecision(
            allowed=False,
            reason=f"Tool '{tool_name}' is not in the allowed tools list.",
        )


# Alias for default MVP policy
DefaultToolAuthorizationPolicy = AllowlistAuthorizationPolicy
