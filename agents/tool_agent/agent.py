import json
import uuid
from typing import Any, Optional, Tuple

from agents.tool_agent.result import AgentResult
from core.authorization.errors import AuthorizationDeniedError
from core.authorization.policy import AuthorizationPolicy
from core.interfaces.tool_registry import ToolNotFoundError, ToolRegistry
from core.logging.collector import EventCollector
from core.logging.events import Event
from core.models.base import ModelAdapter
from core.models.tool_call import ToolCall


class ToolUsingAgent:
    """
    Controlled tool-using agent.

    Coordinates between a ModelAdapter, ToolRegistry, an AuthorizationPolicy,
    and an optional EventCollector for observability.
    Ensures that tool requests from the model are checked against the authorization
    boundary before any tool execution takes place.
    """

    def __init__(
        self,
        model: ModelAdapter,
        tool_registry: ToolRegistry,
        authorization_policy: Optional[AuthorizationPolicy] = None,
        event_collector: Optional[EventCollector] = None,
        system_prompt_version: Optional[str] = None,
    ) -> None:
        """
        Initialize the ToolUsingAgent.

        Args:
            model: The ModelAdapter to generate responses or tool calls.
            tool_registry: The ToolRegistry containing available tools.
            authorization_policy: Optional AuthorizationPolicy to evaluate before tool execution.
            event_collector: Optional EventCollector to record interaction events.
            system_prompt_version: Optional version identifier for system prompts.
        """
        self.model = model
        self.tool_registry = tool_registry
        self.authorization_policy = authorization_policy
        self.event_collector = event_collector
        self.system_prompt_version = system_prompt_version

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

    def _parse_response(self, raw_response: str) -> Tuple[str, Any]:
        """
        Safely parse the model response into a final response or structured tool call.

        Does not use eval() or code execution.
        """
        try:
            data = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError):
            return "final", raw_response

        if not isinstance(data, dict):
            return "final", str(data)

        resp_type = data.get("type")
        if resp_type == "tool_call":
            tool_name = data.get("tool") or data.get("tool_name")
            if not tool_name or not isinstance(tool_name, str):
                raise ValueError("Tool call missing valid 'tool' or 'tool_name' field.")
            arguments = data.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError("Tool call 'arguments' must be a dictionary.")
            return "tool_call", ToolCall(tool_name=tool_name, arguments=arguments)

        if resp_type == "final":
            return "final", str(data.get("content", ""))

        if "tool" in data or "tool_name" in data:
            tool_name = data.get("tool") or data.get("tool_name")
            if not tool_name or not isinstance(tool_name, str):
                raise ValueError("Tool call missing valid 'tool' field.")
            arguments = data.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError("Tool call 'arguments' must be a dictionary.")
            return "tool_call", ToolCall(tool_name=tool_name, arguments=arguments)

        if "content" in data:
            return "final", str(data["content"])

        return "final", raw_response

    def send_message(self, user_input: str, session_id: Optional[str] = None) -> AgentResult:
        """
        Process a user message: send to model, evaluate tool calls with authorization,
        execute if authorized, and return a structured agent result.

        Args:
            user_input: The input message from the user.
            session_id: Optional session identifier for request correlation.

        Returns:
            An AgentResult detailing the final response, tool execution, or authorization denial.
        """
        sid = session_id or uuid.uuid4().hex[:12]

        self._emit_event(
            session_id=sid,
            event_type="user_input",
            agent_state="received_input",
            user_input=user_input,
        )

        try:
            raw_response = self.model.generate(user_input)
        except Exception as e:
            error_msg = f"Model error: {str(e)}"
            self._emit_event(
                session_id=sid,
                event_type="error",
                agent_state="error",
                error=error_msg,
            )
            return AgentResult(session_id=sid, error=error_msg)

        self._emit_event(
            session_id=sid,
            event_type="model_response",
            agent_state="model_responded",
            final_response=raw_response,
        )

        try:
            resp_type, parsed = self._parse_response(raw_response)
        except Exception as e:
            error_msg = f"Parse error: {str(e)}"
            self._emit_event(
                session_id=sid,
                event_type="error",
                agent_state="error",
                error=error_msg,
            )
            return AgentResult(session_id=sid, error=error_msg)

        if resp_type == "final":
            self._emit_event(
                session_id=sid,
                event_type="final_response",
                agent_state="completed",
                final_response=parsed,
            )
            return AgentResult(session_id=sid, final_response=parsed)

        # Handle tool call
        tool_call: ToolCall = parsed
        self._emit_event(
            session_id=sid,
            event_type="tool_call",
            agent_state="tool_requested",
            tool_call=tool_call.tool_name,
            tool_arguments=tool_call.arguments,
        )

        # Evaluate policy before execution if available
        policy = self.authorization_policy or self.tool_registry.authorization_policy
        auth_allowed = False
        auth_reason = None
        if policy is not None:
            decision = policy.evaluate(tool_name=tool_call.tool_name, arguments=tool_call.arguments)
            auth_allowed = decision.allowed
            auth_reason = decision.reason
            self._emit_event(
                session_id=sid,
                event_type="authorization_decision",
                agent_state="authorization_evaluated",
                tool_call=tool_call.tool_name,
                authorization_decision=decision.to_dict(),
            )
            if not decision.allowed:
                self._emit_event(
                    session_id=sid,
                    event_type="authorization_denied",
                    agent_state="authorization_denied",
                    tool_call=tool_call.tool_name,
                    error=decision.reason,
                    security_event="authorization_denied",
                )
                return AgentResult(
                    session_id=sid,
                    tool_call=tool_call,
                    authorization_allowed=False,
                    authorization_denied=True,
                    authorization_reason=decision.reason,
                    tool_executed=False,
                    error=f"Authorization denied for tool '{tool_call.tool_name}': {decision.reason}",
                )

        try:
            result = self.tool_registry.execute(
                name=tool_call.tool_name,
                args=tool_call.arguments,
                authorization_policy=self.authorization_policy,
            )
            self._emit_event(
                session_id=sid,
                event_type="tool_execution",
                agent_state="tool_executed",
                tool_call=tool_call.tool_name,
                tool_result=result,
                execution_result=result,
            )
            self._emit_event(
                session_id=sid,
                event_type="final_response",
                agent_state="completed",
                final_response=str(result),
            )
            return AgentResult(
                session_id=sid,
                tool_call=tool_call,
                authorization_allowed=auth_allowed,
                authorization_denied=False,
                authorization_reason=auth_reason,
                tool_executed=True,
                tool_result=result,
                final_response=str(result),
            )
        except AuthorizationDeniedError as e:
            self._emit_event(
                session_id=sid,
                event_type="authorization_denied",
                agent_state="authorization_denied",
                tool_call=tool_call.tool_name,
                error=e.reason,
                security_event="authorization_denied",
            )
            return AgentResult(
                session_id=sid,
                tool_call=tool_call,
                authorization_allowed=False,
                authorization_denied=True,
                authorization_reason=e.reason,
                tool_executed=False,
                error=str(e),
            )
        except ToolNotFoundError as e:
            error_msg = f"Tool '{tool_call.tool_name}' not found: {str(e)}"
            self._emit_event(
                session_id=sid,
                event_type="error",
                agent_state="error",
                tool_call=tool_call.tool_name,
                error=error_msg,
            )
            return AgentResult(
                session_id=sid,
                tool_call=tool_call,
                authorization_allowed=auth_allowed,
                authorization_denied=False,
                authorization_reason=auth_reason,
                tool_executed=False,
                error=error_msg,
            )
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            self._emit_event(
                session_id=sid,
                event_type="error",
                agent_state="error",
                tool_call=tool_call.tool_name,
                error=error_msg,
            )
            return AgentResult(
                session_id=sid,
                tool_call=tool_call,
                authorization_allowed=auth_allowed,
                authorization_denied=False,
                authorization_reason=auth_reason,
                tool_executed=False,
                error=error_msg,
            )
