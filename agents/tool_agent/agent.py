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
from core.models.tool_schema import PromptBuilder



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
        text = raw_response.strip()
        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        # Check for prose mixed with tool-call JSON
        if not (text.startswith("{") and text.endswith("}")) and not (text.startswith("[") and text.endswith("]")):
            if '"type": "tool_call"' in text or '"tool_call"' in text or '"tool":' in text:
                raise ValueError("Prose mixed with tool-call JSON is not permitted. Response must be strictly a JSON object.")
            return "final", raw_response

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            if text.startswith("{") and ("tool_call" in text or "tool" in text):
                raise ValueError(f"Malformed tool-call JSON: {e}")
            return "final", raw_response

        if not isinstance(data, dict):
            if isinstance(data, list) and any(
                isinstance(item, dict) and ("tool" in item or item.get("type") == "tool_call")
                for item in data
            ):
                raise ValueError("Array of tool calls is not permitted. Response must be a single tool_call object.")
            return "final", str(data)

        resp_type = data.get("type")
        if resp_type == "tool_call":
            tool_name = data.get("tool") or data.get("tool_name")
            if not tool_name or not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError("Tool call missing valid 'tool' or 'tool_name' field.")
            arguments = data.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError("Tool call 'arguments' must be a dictionary.")
            return "tool_call", ToolCall(tool_name=tool_name.strip(), arguments=arguments)

        if resp_type == "final":
            content = data.get("response") if "response" in data else data.get("content", "")
            return "final", str(content)

        if "tool" in data or "tool_name" in data:
            tool_name = data.get("tool") or data.get("tool_name")
            if not tool_name or not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError("Tool call missing valid 'tool' field.")
            arguments = data.get("arguments", {})
            if not isinstance(arguments, dict):
                raise ValueError("Tool call 'arguments' must be a dictionary.")
            return "tool_call", ToolCall(tool_name=tool_name.strip(), arguments=arguments)

        if "response" in data:
            return "final", str(data["response"])

        if "content" in data:
            return "final", str(data["content"])

        return "final", raw_response

    def _format_authorization_denied_response(self, tool_name: str, reason: Optional[str]) -> str:
        """Format a safe, natural-language user-facing response for authorization denials."""
        if reason:
            return f"I couldn't execute '{tool_name}' because authorization was denied: {reason}."
        return f"I couldn't execute '{tool_name}' because authorization was denied by policy."

    def _format_execution_error_response(
        self, tool_name: str, arguments: Dict[str, Any], error: Exception
    ) -> str:
        """
        Format a safe, natural-language user-facing response for tool execution failures,
        without exposing internal host filesystem paths or secrets.
        """
        err_str = str(error)
        err_type = type(error).__name__

        # 1. FileTool sandbox violations
        if "SandboxViolationError" in err_type or "outside the sandbox" in err_str:
            op = arguments.get("operation", "access")
            if op == "read":
                return "I couldn't read that file because the requested path is outside the controlled file sandbox."
            elif op == "write":
                return "I couldn't write to that file because the requested path is outside the controlled file sandbox."
            elif op == "list":
                return "I couldn't list that directory because the requested path is outside the controlled file sandbox."
            return "I couldn't perform that file operation because the requested path is outside the controlled file sandbox."

        # 2. FileTool file not found
        if "FileNotFoundError" in err_type or "File or directory not found" in err_str:
            target = arguments.get("path", "file")
            return f"I couldn't find the requested file or directory '{target}' in the controlled file sandbox."

        # 3. FileTool size limit
        if "FileSizeLimitExceededError" in err_type or ("exceeds" in err_str.lower() and "size" in err_str.lower()):
            return "The requested file operation could not be completed because it exceeds the allowed size limit."

        # 4. Calculator errors (e.g. division by zero)
        if tool_name == "calculator" and ("Division by zero" in err_str or "ZeroDivisionError" in err_type):
            return "I couldn't complete the calculation because division by zero is not allowed."

        # 5. CommandTool errors (e.g. timeout, command not allowed)
        if tool_name == "command_tool":
            if "timed out" in err_str.lower():
                return "The command execution timed out."
            if "not allowed" in err_str.lower():
                cmd = arguments.get("command", "")
                return f"The command '{cmd}' is not in the allowlist of permitted commands."

        # 6. DatabaseTool errors
        if tool_name == "database_tool":
            if "Destructive SQL" in err_str or "blocked" in err_str.lower():
                return "The database operation was blocked because only safe, non-destructive queries are permitted."

        # 7. HTTPTool errors
        if tool_name == "http_tool":
            if "External network access is blocked" in err_str:
                return "The HTTP request was blocked because external network access is not permitted."

        # General fallback: return a concise natural language explanation
        clean_msg = err_str.split("\n")[0].strip()
        return f"I encountered an error while executing '{tool_name}': {clean_msg}."

    def _format_parse_error_response(self, error: Exception) -> str:
        """Format a safe, natural-language user-facing response for model parse errors."""
        return "I couldn't complete the requested operation because the model returned an invalid tool-call format."

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

        prompt_builder = PromptBuilder(tool_registry=self.tool_registry)
        current_prompt = prompt_builder.build_initial_prompt(user_input)
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
                final_resp = self._format_parse_error_response(e)
                self._emit_event(
                    session_id=sid,
                    event_type="final_response",
                    agent_state="completed",
                    final_response=final_resp,
                )
                return AgentResult(
                    session_id=sid,
                    tool_call=last_tool_call,
                    tool_calls=executed_tool_calls,
                    tool_results=executed_tool_results,
                    tool_executed=len(executed_tool_results) > 0,
                    error=error_msg,
                    final_response=final_resp,
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
                    tool_result=executed_tool_results[-1] if executed_tool_results else None,
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
                    final_resp = self._format_authorization_denied_response(tool_call.tool_name, decision.reason)
                    self._emit_event(
                        session_id=sid,
                        event_type="final_response",
                        agent_state="completed",
                        final_response=final_resp,
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
                        tool_result=None,
                        error=f"Authorization denied for tool '{tool_call.tool_name}': {decision.reason}",
                        final_response=final_resp,
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

                current_prompt = prompt_builder.build_feedback_prompt(
                    user_input=user_input,
                    tool_name=tool_call.tool_name,
                    tool_arguments=tool_call.arguments,
                    tool_result=result,
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
                final_resp = self._format_authorization_denied_response(tool_call.tool_name, e.reason)
                self._emit_event(
                    session_id=sid,
                    event_type="final_response",
                    agent_state="completed",
                    final_response=final_resp,
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
                    tool_result=None,
                    error=str(e),
                    final_response=final_resp,
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
                final_resp = f"I couldn't execute the requested tool because '{tool_call.tool_name}' was not found."
                self._emit_event(
                    session_id=sid,
                    event_type="final_response",
                    agent_state="completed",
                    final_response=final_resp,
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
                    tool_result=None,
                    error=error_msg,
                    final_response=final_resp,
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
                final_resp = self._format_execution_error_response(
                    tool_call.tool_name, tool_call.arguments, e
                )
                self._emit_event(
                    session_id=sid,
                    event_type="final_response",
                    agent_state="completed",
                    final_response=final_resp,
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
                    tool_result=None,
                    error=error_msg,
                    final_response=final_resp,
                )

        return AgentResult(
            session_id=sid,
            tool_call=last_tool_call,
            tool_calls=executed_tool_calls,
            tool_results=executed_tool_results,
            tool_executed=len(executed_tool_results) > 0,
            authorization_allowed=last_auth_allowed,
            authorization_reason=last_auth_reason,
            tool_result=executed_tool_results[-1] if executed_tool_results else None,
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

