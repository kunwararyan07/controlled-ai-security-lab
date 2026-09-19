from typing import Any, Dict, List, Optional
from core.authorization.errors import AuthorizationDeniedError
from core.authorization.policy import AuthorizationPolicy
from core.interfaces.tool import Tool


class DuplicateToolError(ValueError):
    """Raised when attempting to register a tool with a name that already exists."""
    pass


class ToolNotFoundError(KeyError, ValueError):
    """Raised when requesting or executing an unregistered tool."""
    pass


class ToolRegistry:
    """
    Registry for managing and executing tools.

    Provides registration, lookup, listing, and execution of tools.
    Supports evaluating an AuthorizationPolicy prior to tool execution.
    """

    def __init__(self, authorization_policy: Optional[AuthorizationPolicy] = None) -> None:
        """
        Initialize the ToolRegistry.

        Args:
            authorization_policy: Optional default authorization policy applied to executions.
        """
        self._tools: Dict[str, Tool] = {}
        self._authorization_policy = authorization_policy

    @property
    def authorization_policy(self) -> Optional[AuthorizationPolicy]:
        """Return the default authorization policy for the registry."""
        return self._authorization_policy

    @authorization_policy.setter
    def authorization_policy(self, policy: Optional[AuthorizationPolicy]) -> None:
        """Set the default authorization policy for the registry."""
        self._authorization_policy = policy

    def register(self, tool: Tool) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: A Tool instance to register.

        Raises:
            TypeError: If the tool does not inherit from Tool.
            ValueError: If tool name is invalid.
            DuplicateToolError: If a tool with the same name is already registered.
        """
        if not isinstance(tool, Tool):
            raise TypeError(f"Expected Tool instance, got {type(tool).__name__}")
        name = tool.name
        if not name or not isinstance(name, str):
            raise ValueError("Tool name must be a non-empty string.")
        if name in self._tools:
            raise DuplicateToolError(f"Tool '{name}' is already registered.")
        self._tools[name] = tool

    def get(self, name: str) -> Tool:
        """
        Retrieve a registered tool by name.

        Args:
            name: The name of the tool to retrieve.

        Returns:
            The registered Tool instance.

        Raises:
            ToolNotFoundError: If the tool is not registered.
        """
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' is not registered.")
        return self._tools[name]

    def list_tools(self) -> List[str]:
        """
        List the names of all registered tools.

        Returns:
            A list of registered tool names.
        """
        return list(self._tools.keys())

    def get_tool_metadata(self) -> List[Dict[str, Any]]:
        """
        Retrieve structured metadata for all registered tools.

        Returns:
            A list of metadata dictionaries for each registered tool.
        """
        return [tool.get_metadata() for tool in self._tools.values()]

    def execute(
        self,
        name: str,
        args: Optional[Dict[str, Any]] = None,
        authorization_policy: Optional[AuthorizationPolicy] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a registered tool by name with structured arguments and authorization.

        Args:
            name: The name of the tool to execute.
            args: Optional structured dictionary of input arguments.
            authorization_policy: Optional policy to evaluate before execution.
                Overrides any default policy set on the registry.
            context: Optional contextual information passed to the authorization policy.
            **kwargs: Additional keyword arguments forwarded to the tool.

        Returns:
            The result of the tool execution.

        Raises:
            ToolNotFoundError: If the tool is not registered.
            AuthorizationDeniedError: If the execution is denied by the authorization policy.
        """
        # 1. Resolve tool (raises ToolNotFoundError if not registered)
        tool = self.get(name)

        # 2. Determine applicable authorization policy
        policy = authorization_policy if authorization_policy is not None else self._authorization_policy

        # 3. Check authorization before executing
        if policy is not None:
            call_args: Dict[str, Any] = {}
            if args is not None and isinstance(args, dict):
                call_args.update(args)
            call_args.update(kwargs)
            eval_args = call_args if (call_args or args is None) else args

            decision = policy.evaluate(tool_name=name, arguments=eval_args, context=context)
            if not decision.allowed:
                raise AuthorizationDeniedError(tool_name=name, reason=decision.reason)

        # 4. Execute tool only after authorization succeeds
        return tool.execute(args, **kwargs)
