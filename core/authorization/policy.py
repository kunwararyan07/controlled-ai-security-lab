from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple


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

    @property
    def is_allowed(self) -> bool:
        return self.allowed

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

    def authorize(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthorizationDecision:
        """Convenience alias for evaluate()."""
        return self.evaluate(tool_name, arguments=arguments, context=context)


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


@dataclass
class ToolScopeConstraint:
    """
    Scope constraint definition for a specific tool.

    Why scope-aware authorization is required for ASI02 (Tool Misuse & Exploitation):
    Coarse tool-level allowlists only verify *which* tool is requested. They cannot prevent
    or detect when a legitimate, authorized tool is invoked with out-of-scope, unauthorized,
    or dangerous arguments (e.g., accessing unauthorized filesystem paths, querying sensitive
    database tables, or targeting unauthorized network endpoints). Scope-aware authorization
    enforces least-privilege boundaries at the argument and resource level, which is central
    to evaluating ASI02 defenses.
    """

    allowed_operations: Optional[Set[str]] = None
    disallowed_operations: Optional[Set[str]] = None
    allowed_paths: Optional[Set[str]] = None
    disallowed_paths: Optional[Set[str]] = None
    allowed_path_prefixes: Optional[List[str]] = None
    allowed_commands: Optional[Set[str]] = None
    disallowed_commands: Optional[Set[str]] = None
    allowed_methods: Optional[Set[str]] = None
    allowed_endpoints: Optional[Set[str]] = None
    disallowed_endpoints: Optional[Set[str]] = None
    allowed_recipients: Optional[Set[str]] = None
    disallowed_recipients: Optional[Set[str]] = None
    allowed_tables: Optional[Set[str]] = None
    disallowed_tables: Optional[Set[str]] = None
    custom_validator: Optional[Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Tuple[bool, str]]] = None

    def evaluate(
        self,
        arguments: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Evaluate structured arguments against configured constraints.

        Returns:
            Tuple of (allowed: bool, reason: str).
        """
        args = arguments or {}

        # 1. Operation constraints (file_tool, database_tool, notification_tool, calculator)
        op = args.get("operation")
        if op is not None and isinstance(op, str):
            op_norm = op.strip().lower()
            if self.allowed_operations is not None and op_norm not in {o.lower() for o in self.allowed_operations}:
                return False, f"Operation '{op}' is not in allowed operations: {sorted(list(self.allowed_operations))}."
            if self.disallowed_operations is not None and op_norm in {o.lower() for o in self.disallowed_operations}:
                return False, f"Operation '{op}' is explicitly disallowed by scope policy."

        # 2. Path constraints (file_tool)
        path = args.get("path") or args.get("filepath") or args.get("file_path")
        if path is not None and isinstance(path, str):
            clean_path = path.strip()
            if self.allowed_paths is not None and "*" not in self.allowed_paths:
                in_allowed = (clean_path in self.allowed_paths) or any(
                    clean_path.startswith(ap) for ap in self.allowed_paths if ap.endswith("/")
                )
                if not in_allowed:
                    return False, f"Path '{clean_path}' is not within permitted paths: {sorted(list(self.allowed_paths))}."
            if self.disallowed_paths is not None:
                in_disallowed = (clean_path in self.disallowed_paths) or any(
                    clean_path.startswith(dp) for dp in self.disallowed_paths if dp.endswith("/")
                )
                if in_disallowed:
                    return False, f"Path '{clean_path}' is explicitly disallowed by scope policy."
            if self.allowed_path_prefixes is not None:
                if not any(clean_path.startswith(prefix) for prefix in self.allowed_path_prefixes):
                    return False, f"Path '{clean_path}' does not match any allowed prefix: {self.allowed_path_prefixes}."

        # 3. Command constraints (command_tool)
        cmd = args.get("command")
        if cmd is not None and isinstance(cmd, str):
            cmd_norm = cmd.strip().lower()
            if self.allowed_commands is not None and cmd_norm not in {c.lower() for c in self.allowed_commands}:
                return False, f"Command '{cmd}' is not in allowed commands: {sorted(list(self.allowed_commands))}."
            if self.disallowed_commands is not None and cmd_norm in {c.lower() for c in self.disallowed_commands}:
                return False, f"Command '{cmd}' is explicitly disallowed by scope policy."

        # 4. HTTP method and endpoint constraints (http_tool)
        method = args.get("method")
        if method is not None and isinstance(method, str):
            m_norm = method.strip().upper()
            if self.allowed_methods is not None and m_norm not in {m.upper() for m in self.allowed_methods}:
                return False, f"HTTP method '{method}' is not in allowed methods: {sorted(list(self.allowed_methods))}."

        endpoint = args.get("endpoint") or args.get("url") or (args.get("path") if method is not None else None)
        if endpoint is not None and isinstance(endpoint, str):
            ep_norm = endpoint.strip().split("?")[0].rstrip("/") or "/"
            if self.allowed_endpoints is not None and "*" not in self.allowed_endpoints:
                normalized_allowed = {e.strip().split("?")[0].rstrip("/") or "/" for e in self.allowed_endpoints}
                in_allowed_ep = (ep_norm in normalized_allowed) or any(
                    ep_norm.startswith(ae) for ae in normalized_allowed if ae.endswith("/") or ae.startswith("http")
                )
                if not in_allowed_ep:
                    return False, f"HTTP endpoint '{endpoint}' is not in allowed endpoints: {sorted(list(self.allowed_endpoints))}."
            if self.disallowed_endpoints is not None:
                normalized_disallowed = {e.strip().split("?")[0].rstrip("/") or "/" for e in self.disallowed_endpoints}
                in_disallowed_ep = (ep_norm in normalized_disallowed) or any(
                    ep_norm.startswith(de) for de in normalized_disallowed if de.endswith("/") or de.startswith("http")
                )
                if in_disallowed_ep:
                    return False, f"HTTP endpoint '{endpoint}' is explicitly disallowed by scope policy."

        # 5. Recipient constraints (notification_tool)
        recipient = args.get("recipient")
        if recipient is not None and isinstance(recipient, str):
            rec_norm = recipient.strip().lower()
            if self.allowed_recipients is not None and rec_norm not in {r.lower() for r in self.allowed_recipients}:
                return False, f"Recipient '{recipient}' is not in allowed recipients: {sorted(list(self.allowed_recipients))}."
            if self.disallowed_recipients is not None and rec_norm in {r.lower() for r in self.disallowed_recipients}:
                return False, f"Recipient '{recipient}' is explicitly disallowed by scope policy."

        # 6. Database table constraints (database_tool)
        query = args.get("query")
        if query is not None and isinstance(query, str):
            if self.disallowed_tables is not None:
                for dt in self.disallowed_tables:
                    if re.search(rf"\b{re.escape(dt)}\b", query, re.IGNORECASE):
                        return False, f"Query references disallowed table '{dt}'."

        # 7. Custom validator
        if self.custom_validator is not None:
            try:
                c_allowed, c_reason = self.custom_validator(args, context)
            except TypeError:
                c_allowed, c_reason = self.custom_validator(args)
            if not c_allowed:
                return False, c_reason

        return True, "Arguments satisfy scope constraints."

    def validate(
        self,
        arguments: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Convenience alias for evaluate()."""
        return self.evaluate(arguments=arguments, context=context)


class ScopeAuthorizationPolicy(AllowlistAuthorizationPolicy):
    """
    Scope-aware authorization policy that enforces both tool-level and argument-level controls.

    Why scope-aware authorization is required for ASI02:
    In ASI02 (Tool Misuse & Exploitation), the model often has legitimate access to a tool,
    but is tricked or confused into calling it with unauthorized parameters (e.g. accessing
    restricted files or executing unintended operations). Coarse allowlists cannot prevent this.
    ScopeAuthorizationPolicy enforces fine-grained argument and resource constraints.
    """

    def __init__(
        self,
        allowed_tools: Optional[Iterable[str]] = None,
        constraints: Optional[Dict[str, ToolScopeConstraint]] = None,
        tool_constraints: Optional[Dict[str, ToolScopeConstraint]] = None,
    ) -> None:
        super().__init__(allowed_tools=allowed_tools)
        resolved = constraints if constraints is not None else tool_constraints
        self.constraints: Dict[str, ToolScopeConstraint] = dict(resolved) if resolved is not None else {}

    def set_constraint(self, tool_name: str, constraint: ToolScopeConstraint) -> None:
        """Associate a scope constraint with a specific tool name."""
        self.constraints[tool_name] = constraint

    def remove_constraint(self, tool_name: str) -> None:
        """Remove scope constraints for a specific tool name."""
        self.constraints.pop(tool_name, None)

    def add_scope(
        self,
        tool_name: str,
        allowed_operations: Optional[Iterable[str]] = None,
        disallowed_operations: Optional[Iterable[str]] = None,
        allowed_paths: Optional[Iterable[str]] = None,
        disallowed_paths: Optional[Iterable[str]] = None,
        allowed_path_prefixes: Optional[List[str]] = None,
        allowed_commands: Optional[Iterable[str]] = None,
        disallowed_commands: Optional[Iterable[str]] = None,
        allowed_methods: Optional[Iterable[str]] = None,
        allowed_endpoints: Optional[Iterable[str]] = None,
        disallowed_endpoints: Optional[Iterable[str]] = None,
        allowed_recipients: Optional[Iterable[str]] = None,
        disallowed_recipients: Optional[Iterable[str]] = None,
        allowed_tables: Optional[Iterable[str]] = None,
        disallowed_tables: Optional[Iterable[str]] = None,
        custom_validator: Optional[Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Tuple[bool, str]]] = None,
    ) -> None:
        """Convenience method to register a tool and configure its scope constraints."""
        self.add_tool(tool_name)
        constraint = ToolScopeConstraint(
            allowed_operations=set(allowed_operations) if allowed_operations is not None else None,
            disallowed_operations=set(disallowed_operations) if disallowed_operations is not None else None,
            allowed_paths=set(allowed_paths) if allowed_paths is not None else None,
            disallowed_paths=set(disallowed_paths) if disallowed_paths is not None else None,
            allowed_path_prefixes=list(allowed_path_prefixes) if allowed_path_prefixes is not None else None,
            allowed_commands=set(allowed_commands) if allowed_commands is not None else None,
            disallowed_commands=set(disallowed_commands) if disallowed_commands is not None else None,
            allowed_methods=set(allowed_methods) if allowed_methods is not None else None,
            allowed_endpoints=set(allowed_endpoints) if allowed_endpoints is not None else None,
            disallowed_endpoints=set(disallowed_endpoints) if disallowed_endpoints is not None else None,
            allowed_recipients=set(allowed_recipients) if allowed_recipients is not None else None,
            disallowed_recipients=set(disallowed_recipients) if disallowed_recipients is not None else None,
            allowed_tables=set(allowed_tables) if allowed_tables is not None else None,
            disallowed_tables=set(disallowed_tables) if disallowed_tables is not None else None,
            custom_validator=custom_validator,
        )
        self.set_constraint(tool_name, constraint)

    def evaluate(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthorizationDecision:
        # Step 1: Check tool allowlist
        if tool_name not in self.allowed_tools:
            return AuthorizationDecision(
                allowed=False,
                reason=f"Tool '{tool_name}' is not in the allowed tools list.",
            )

        # Step 2: Check tool scope constraint if defined
        constraint = self.constraints.get(tool_name)
        if constraint is not None:
            allowed, reason = constraint.evaluate(arguments=arguments, context=context)
            if not allowed:
                return AuthorizationDecision(
                    allowed=False,
                    reason=f"Scope authorization denied for tool '{tool_name}': {reason}",
                )

        return AuthorizationDecision(
            allowed=True,
            reason=f"Tool '{tool_name}' is permitted by scope policy.",
        )


# Alias for default MVP policy
DefaultToolAuthorizationPolicy = AllowlistAuthorizationPolicy
