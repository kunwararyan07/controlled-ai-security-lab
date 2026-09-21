from typing import Any, Dict, Optional
from core.interfaces.tool import Tool


class CalculatorError(Exception):
    """Base exception for calculator tool errors."""
    pass


class InvalidOperationError(CalculatorError, ValueError, KeyError):
    """Raised when an invalid or unsupported operation is requested."""
    pass


class DivisionByZeroError(CalculatorError, ZeroDivisionError, ValueError):
    """Raised when division by zero is attempted."""
    pass


class InvalidArgumentError(CalculatorError, ValueError, TypeError):
    """Raised when arguments are invalid, missing, or not valid numbers."""
    pass


class CalculatorTool(Tool):
    """
    Deterministic calculator tool for basic arithmetic operations.

    Executes addition, subtraction, multiplication, and division safely
    using structured numeric parameters without using eval() or code execution.
    """

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Performs basic arithmetic operations using structured numeric inputs (operation, a, b). "
            "'operation' is required. Allowed operation names are exactly: add, subtract, multiply, divide. "
            "'a' is required and must be numeric. 'b' is required and must be numeric. "
            "The model MUST NOT use an 'expression' field and must not invent alternate formats "
            'such as {"expression":"25 * 4"}. '
            "The calculator interface is operation/a/b and all arithmetic expressions must be converted "
            "into that structure. "
            "Examples: "
            'Addition: {"operation":"add","a":10,"b":5}; '
            'Subtraction: {"operation":"subtract","a":10,"b":5}; '
            'Multiplication: {"operation":"multiply","a":25,"b":4}; '
            'Division: {"operation":"divide","a":20,"b":5}.'
        )

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": (
                        "The arithmetic operation to perform. Required. "
                        "Allowed operation names are exactly: add, subtract, multiply, divide."
                    ),
                },
                "a": {
                    "type": "number",
                    "description": "The first operand. Required and must be numeric.",
                },
                "b": {
                    "type": "number",
                    "description": "The second operand. Required and must be numeric.",
                },
            },
            "required": ["operation", "a", "b"],
        }

    def execute(self, args: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """
        Execute the arithmetic operation safely with input validation.

        Args:
            args: Dictionary containing 'operation', 'a', and 'b'.
            **kwargs: Additional keyword arguments forwarded or passed directly.

        Returns:
            The numeric result of the arithmetic operation.

        Raises:
            InvalidArgumentError: If arguments are missing or not valid numbers.
            InvalidOperationError: If the operation is not supported.
            DivisionByZeroError: If division by zero is attempted.
        """
        if args is not None:
            if not isinstance(args, dict):
                raise InvalidArgumentError("Arguments must be provided as a dictionary.")
            params = dict(args)
            params.update(kwargs)
        else:
            params = kwargs

        if not params:
            raise InvalidArgumentError("Missing arguments for calculator.")

        if "operation" not in params:
            raise InvalidArgumentError("Missing required argument: 'operation'.")

        operation = params["operation"]
        if not isinstance(operation, str):
            raise InvalidArgumentError("Argument 'operation' must be a string.")

        op = operation.strip().lower()

        if "a" not in params:
            raise InvalidArgumentError("Missing required argument: 'a'.")
        if "b" not in params:
            raise InvalidArgumentError("Missing required argument: 'b'.")

        a = params["a"]
        b = params["b"]

        # Disallow boolean types (since bool is a subclass of int in Python)
        if isinstance(a, bool) or not isinstance(a, (int, float)):
            raise InvalidArgumentError(
                f"Argument 'a' must be an integer or float, got {type(a).__name__}."
            )
        if isinstance(b, bool) or not isinstance(b, (int, float)):
            raise InvalidArgumentError(
                f"Argument 'b' must be an integer or float, got {type(b).__name__}."
            )

        if op == "add":
            return a + b
        elif op == "subtract":
            return a - b
        elif op == "multiply":
            return a * b
        elif op == "divide":
            if b == 0:
                raise DivisionByZeroError("Division by zero is not allowed.")
            return a / b
        else:
            raise InvalidOperationError(
                f"Unsupported operation: '{operation}'. Supported operations are: add, subtract, multiply, divide."
            )


# Alias for convenience
Calculator = CalculatorTool
