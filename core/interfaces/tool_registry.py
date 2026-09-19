from typing import Any, Dict, List, Optional
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
    Designed to allow future authorization and observability layers
    to wrap around tool execution cleanly.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

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

    def execute(self, name: str, args: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """
        Execute a registered tool by name with structured arguments.

        Args:
            name: The name of the tool to execute.
            args: Optional structured dictionary of input arguments.
            **kwargs: Additional keyword arguments forwarded to the tool.

        Returns:
            The result of the tool execution.

        Raises:
            ToolNotFoundError: If the tool is not registered.
        """
        tool = self.get(name)
        # Future authorization and observability hooks can wrap tool execution here.
        return tool.execute(args, **kwargs)
