import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

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
        max_steps: int = 1,
    ) -> None:
        """
        Initialize the ToolUsingAgent.

        Args:
            model: The ModelAdapter to generate responses or tool calls.
            tool_registry: The ToolRegistry containing available tools.
            authorization_policy: Optional AuthorizationPolicy to evaluate before tool execution.
            event_collector: Optional EventCollector to record interaction events.
            system_prompt_version: Optional version identifier for system prompts.
            max_steps: Maximum execution steps for multi-step tool chains (default: 1).
        """
        self.model = model
        self.tool_registry = tool_registry
        self.authorization_policy = authorization_policy
        self.event_collector = event_collector
        self.system_prompt_version = system_prompt_version
        self.max_steps = max_steps
        self._state: str = "idle"
        self._is_running: bool = True

    def start(self) -> "ToolUsingAgent":
        """Start the agent and set state to running."""
        self._is_running = True
        self._state = "running"
        return self

    def stop(self) -> "ToolUsingAgent":
        """Stop the agent and set state to stopped."""
        self._is_running = False
        self._state = "stopped"
        return self

    def reset(self) -> None:
        """
        Reset agent lifecycle state, clear event collector logs,
        and reset all registered tools that provide a reset() method.
        """
        self._state = "idle"
        if self.event_collector is not None:
            self.event_collector.clear()
        if self.tool_registry is not None:
            for tool_name in self.tool_registry.list_tools():
                tool_instance = self.tool_registry.get(tool_name)
                if hasattr(tool_instance, "reset") and callable(tool_instance.reset):
                    tool_instance.reset()

    def get_state(self) -> Dict[str, Any]:
        """Return the current agent lifecycle and configuration state."""
        return {
            "state": self._state,
            "is_running": self._is_running,
            "tools": self.tool_registry.list_tools() if self.tool_registry else [],
            "has_authorization_policy": self.authorization_policy is not None,
            "has_event_collector": self.event_collector is not None,
            "system_prompt_version": self.system_prompt_version,
            "max_steps": self.max_steps,
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

    def send_message(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        max_steps: Optional[int] = None,
    ) -> AgentResult:
        """
        Process a user message: send to model, evaluate tool calls with authorization,
        execute if authorized, and return a structured agent result. Supports single-step
        and multi-step tool execution chains.

        Args:
            user_input: The input message from the user.
            session_id: Optional session identifier for request correlation.
            max_steps: Optional override for maximum execution steps in a multi-step chain.

        Returns:
            An AgentResult detailing final response, tool execution(s), or authorization denial.
        """
        sid = session_id or uuid.uuid4().hex[:12]
        effective_max_steps = max_steps if max_steps is not None else self.max_steps
        if effective_max_steps < 1:
            effective_max_steps = 1

        self._emit_event(
            session_id=sid,
            event_type="user_input",
            agent_state="received_input",
            user_input=user_input,
        )

        current_prompt = user_input
        executed_tool_calls: List[ToolCall] = []
        executed_tool_results: List[Any] = []
        last_tool_call: Optional[ToolCall] = None
        last_auth_allowed: bool = False
        last_auth_reason: Optional[str] = None

        step = 0
        while step < effective_max_steps:
            step += 1

            try:
                raw_response = self.model.generate(current_prompt)
            except Exception as e:
                error_msg = f"Model error: {str(e)}"
                self._emit_event(
                    session_id=sid,
                    event_type="error",
                    agent_state="error",
                    error=error_msg,
                )
                return AgentResult(
                    session_id=sid,
                    tool_call=last_tool_call,
                    tool_calls=executed_tool_calls,
                    tool_results=executed_tool_results,
                    tool_executed=len(executed_tool_results) > 0,
                    error=error_msg,
                )

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
                return AgentResult(
                    session_id=sid,
                    tool_call=last_tool_call,
                    tool_calls=executed_tool_calls,
                    tool_results=executed_tool_results,
                    tool_executed=len(executed_tool_results) > 0,
                    error=error_msg,
                )

            if resp_type == "final":
                self._emit_event(
                    session_id=sid,
                    event_type="final_response",
                    agent_state="completed",
                    final_response=parsed,
                )
                return AgentResult(
                    session_id=sid,
                    final_response=parsed,
                    tool_call=last_tool_call,
                    tool_calls=executed_tool_calls,
                    tool_results=executed_tool_results,
                    tool_executed=len(executed_tool_results) > 0,
                    authorization_allowed=last_auth_allowed,
                    authorization_reason=last_auth_reason,
                )

            # Handle tool call
            tool_call: ToolCall = parsed
            last_tool_call = tool_call
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
                last_auth_allowed = auth_allowed
                last_auth_reason = auth_reason
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
                        tool_calls=executed_tool_calls + [tool_call],
                        tool_results=executed_tool_results,
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
                executed_tool_calls.append(tool_call)
                executed_tool_results.append(result)
                self._emit_event(
                    session_id=sid,
                    event_type="tool_execution",
                    agent_state="tool_executed",
                    tool_call=tool_call.tool_name,
                    tool_result=result,
                    execution_result=result,
                )

                if step >= effective_max_steps:
                    self._emit_event(
                        session_id=sid,
                        event_type="final_response",
                        agent_state="completed",
                        final_response=str(result),
                    )
                    return AgentResult(
                        session_id=sid,
                        tool_call=tool_call,
                        tool_calls=executed_tool_calls,
                        tool_results=executed_tool_results,
                        authorization_allowed=auth_allowed,
                        authorization_denied=False,
                        authorization_reason=auth_reason,
                        tool_executed=True,
                        tool_result=result,
                        final_response=str(result),
                    )

                current_prompt = str(result)

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
                    tool_calls=executed_tool_calls + [tool_call],
                    tool_results=executed_tool_results,
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
                    tool_calls=executed_tool_calls + [tool_call],
                    tool_results=executed_tool_results,
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
                    tool_calls=executed_tool_calls + [tool_call],
                    tool_results=executed_tool_results,
                    authorization_allowed=auth_allowed,
                    authorization_denied=False,
                    authorization_reason=auth_reason,
                    tool_executed=False,
                    error=error_msg,
                )

        return AgentResult(
            session_id=sid,
            tool_call=last_tool_call,
            tool_calls=executed_tool_calls,
            tool_results=executed_tool_results,
            tool_executed=len(executed_tool_results) > 0,
            authorization_allowed=last_auth_allowed,
            authorization_reason=last_auth_reason,
            final_response=str(executed_tool_results[-1]) if executed_tool_results else None,
        )


def create_full_tool_registry(
    file_tool: Optional[Any] = None,
    database_tool: Optional[Any] = None,
    http_tool: Optional[Any] = None,
    notification_tool: Optional[Any] = None,
    command_tool: Optional[Any] = None,
    authorization_policy: Optional[AuthorizationPolicy] = None,
) -> ToolRegistry:
    """
    Create and return a ToolRegistry populated with all five controlled tools.

    Allows passing custom pre-configured or tracking tool instances, or instantiates
    default controlled tool instances if not provided.

    Args:
        file_tool: Optional FileTool instance.
        database_tool: Optional DatabaseTool instance.
        http_tool: Optional HTTPTool instance.
        notification_tool: Optional NotificationTool instance.
        command_tool: Optional CommandTool instance.
        authorization_policy: Optional default AuthorizationPolicy for the registry.

    Returns:
        A populated ToolRegistry containing all five tools.
    """
    from tools.file_tool import FileTool
    from tools.database_tool import DatabaseTool
    from tools.http_tool import HTTPTool
    from tools.notification_tool import NotificationTool
    from tools.command_tool import CommandTool

    registry = ToolRegistry(authorization_policy=authorization_policy)
    registry.register(file_tool or FileTool())
    registry.register(database_tool or DatabaseTool())
    registry.register(http_tool or HTTPTool())
    registry.register(notification_tool or NotificationTool())
    registry.register(command_tool or CommandTool())
    return registry

