import json
from typing import Any, Optional, Tuple

from agents.tool_agent.result import AgentResult
from core.authorization.errors import AuthorizationDeniedError
from core.authorization.policy import AuthorizationPolicy
from core.interfaces.tool_registry import ToolNotFoundError, ToolRegistry
from core.models.base import ModelAdapter
from core.models.tool_call import ToolCall


class ToolUsingAgent:
    """
    Controlled tool-using agent.

    Coordinates between a ModelAdapter, ToolRegistry, and an AuthorizationPolicy.
    Ensures that tool requests from the model are checked against the authorization
    boundary before any tool execution takes place.
    """

    def __init__(
        self,
        model: ModelAdapter,
        tool_registry: ToolRegistry,
        authorization_policy: Optional[AuthorizationPolicy] = None,
    ) -> None:
        """
        Initialize the ToolUsingAgent.

        Args:
            model: The ModelAdapter to generate responses or tool calls.
            tool_registry: The ToolRegistry containing available tools.
            authorization_policy: Optional AuthorizationPolicy to evaluate before tool execution.
        """
        self.model = model
        self.tool_registry = tool_registry
        self.authorization_policy = authorization_policy

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

    def send_message(self, user_input: str) -> AgentResult:
        """
        Process a user message: send to model, evaluate tool calls with authorization,
        execute if authorized, and return a structured agent result.

        Args:
            user_input: The input message from the user.

        Returns:
            An AgentResult detailing the final response, tool execution, or authorization denial.
        """
        try:
            raw_response = self.model.generate(user_input)
        except Exception as e:
            return AgentResult(error=f"Model error: {str(e)}")

        try:
            resp_type, parsed = self._parse_response(raw_response)
        except Exception as e:
            return AgentResult(error=f"Parse error: {str(e)}")

        if resp_type == "final":
            return AgentResult(final_response=parsed)

        # Handle tool call
        tool_call: ToolCall = parsed

        try:
            result = self.tool_registry.execute(
                name=tool_call.tool_name,
                args=tool_call.arguments,
                authorization_policy=self.authorization_policy,
            )
            return AgentResult(
                tool_call=tool_call,
                tool_executed=True,
                tool_result=result,
                final_response=str(result),
            )
        except AuthorizationDeniedError as e:
            return AgentResult(
                tool_call=tool_call,
                authorization_denied=True,
                authorization_reason=e.reason,
                tool_executed=False,
                error=str(e),
            )
        except ToolNotFoundError as e:
            return AgentResult(
                tool_call=tool_call,
                tool_executed=False,
                error=f"Tool '{tool_call.tool_name}' not found: {str(e)}",
            )
        except Exception as e:
            return AgentResult(
                tool_call=tool_call,
                tool_executed=False,
                error=f"Tool execution failed: {str(e)}",
            )
