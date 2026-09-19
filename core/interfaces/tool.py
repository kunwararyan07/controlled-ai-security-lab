from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Tool(ABC):
    """
    Abstract base interface for all tools used in the security lab.

    Provides a provider-independent contract exposing tool metadata,
    input schema, and structured execution.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable unique identifier for the tool."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        raise NotImplementedError

    @property
    def schema(self) -> Dict[str, Any]:
        """
        Expected input arguments schema for the tool.
        Subclasses may override this to define JSON-schema-compatible parameters.
        """
        return {}

    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Structured metadata for the tool, including name, description, and schema.
        """
        return {
            "name": self.name,
            "description": self.description,
            "schema": self.schema,
        }

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return structured metadata for the tool.
        """
        return self.metadata

    @abstractmethod
    def execute(self, args: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """
        Execute the tool with structured arguments.

        Args:
            args: Structured dictionary of input arguments.
            **kwargs: Additional keyword arguments for flexible execution.

        Returns:
            The execution result.
        """
        raise NotImplementedError
