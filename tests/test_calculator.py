import unittest

from core.interfaces.tool import Tool
from core.models.tool_schema import generate_tool_schema
from tools.calculator import (
    Calculator,
    CalculatorTool,
    DivisionByZeroError,
    InvalidArgumentError,
    InvalidOperationError,
)


class TestCalculatorTool(unittest.TestCase):
    def setUp(self):
        self.calculator = CalculatorTool()

    def test_conforms_to_tool_interface(self):
        self.assertIsInstance(self.calculator, Tool)
        self.assertEqual(self.calculator.name, "calculator")
        self.assertIsInstance(self.calculator.description, str)
        self.assertTrue(len(self.calculator.description) > 0)
        self.assertIn("properties", self.calculator.schema)
        metadata = self.calculator.get_metadata()
        self.assertEqual(metadata["name"], "calculator")
        self.assertEqual(metadata["description"], self.calculator.description)
        self.assertEqual(metadata["schema"], self.calculator.schema)

    def test_addition(self):
        result = self.calculator.execute({"operation": "add", "a": 10, "b": 5})
        self.assertEqual(result, 15)

        # Test negative numbers
        self.assertEqual(self.calculator.execute({"operation": "add", "a": -5, "b": 3}), -2)

        # Test floating point addition
        self.assertAlmostEqual(self.calculator.execute({"operation": "add", "a": 1.5, "b": 2.5}), 4.0)

    def test_subtraction(self):
        result = self.calculator.execute({"operation": "subtract", "a": 10, "b": 5})
        self.assertEqual(result, 5)

        # Negative result
        self.assertEqual(self.calculator.execute({"operation": "subtract", "a": 3, "b": 7}), -4)

    def test_multiplication(self):
        result = self.calculator.execute({"operation": "multiply", "a": 10, "b": 5})
        self.assertEqual(result, 50)

        # Multiply by zero
        self.assertEqual(self.calculator.execute({"operation": "multiply", "a": 10, "b": 0}), 0)

    def test_division(self):
        result = self.calculator.execute({"operation": "divide", "a": 10, "b": 5})
        self.assertEqual(result, 2.0)

        # Non-integer division
        self.assertAlmostEqual(self.calculator.execute({"operation": "divide", "a": 7, "b": 2}), 3.5)

    def test_division_by_zero_handling(self):
        with self.assertRaises(ZeroDivisionError):
            self.calculator.execute({"operation": "divide", "a": 10, "b": 0})

        with self.assertRaises(DivisionByZeroError):
            self.calculator.execute({"operation": "divide", "a": 10, "b": 0})

    def test_invalid_operation_handling(self):
        with self.assertRaises(ValueError):
            self.calculator.execute({"operation": "modulo", "a": 10, "b": 3})

        with self.assertRaises(InvalidOperationError):
            self.calculator.execute({"operation": "unknown_op", "a": 1, "b": 2})

    def test_invalid_or_missing_arguments(self):
        # Empty args
        with self.assertRaises(InvalidArgumentError):
            self.calculator.execute({})

        # Missing operation
        with self.assertRaises(InvalidArgumentError):
            self.calculator.execute({"a": 10, "b": 5})

        # Missing operand a
        with self.assertRaises(InvalidArgumentError):
            self.calculator.execute({"operation": "add", "b": 5})

        # Missing operand b
        with self.assertRaises(InvalidArgumentError):
            self.calculator.execute({"operation": "add", "a": 10})

        # Invalid type for args
        with self.assertRaises((InvalidArgumentError, TypeError)):
            self.calculator.execute("not a dict")

        # Invalid operand types
        with self.assertRaises(InvalidArgumentError):
            self.calculator.execute({"operation": "add", "a": "10", "b": 5})

        with self.assertRaises(InvalidArgumentError):
            self.calculator.execute({"operation": "add", "a": True, "b": 5})

        with self.assertRaises(InvalidArgumentError):
            self.calculator.execute({"operation": "add", "a": 10, "b": [5]})

    def test_arbitrary_code_execution_prevention(self):
        # Malicious strings in operation
        with self.assertRaises(ValueError):
            self.calculator.execute({"operation": "__import__('os').system('ls')", "a": 1, "b": 2})

        # Malicious strings in operands
        with self.assertRaises(ValueError):
            self.calculator.execute({"operation": "add", "a": "__import__('os').system('ls')", "b": 2})

        # Expression string should not be evaluated
        with self.assertRaises(ValueError):
            self.calculator.execute({"operation": "add", "a": "10 + 5", "b": 2})

    def test_deterministic_output(self):
        for _ in range(10):
            self.assertEqual(self.calculator.execute({"operation": "add", "a": 42, "b": 58}), 100)
            self.assertEqual(self.calculator.execute({"operation": "multiply", "a": 6, "b": 7}), 42)

    def test_keyword_arguments_execution(self):
        result = self.calculator.execute(operation="add", a=10, b=5)
        self.assertEqual(result, 15)

    def test_calculator_schema_properties_and_required(self):
        """Regression: Calculator schema specifies operation, a, b and no expression parameter."""
        schema = self.calculator.schema
        self.assertIn("properties", schema)
        self.assertIn("required", schema)

        # Check required fields
        self.assertEqual(sorted(schema["required"]), ["a", "b", "operation"])

        # Check properties
        props = schema["properties"]
        self.assertIn("operation", props)
        self.assertIn("a", props)
        self.assertIn("b", props)
        self.assertNotIn("expression", props)

        # Check allowed operations in schema
        self.assertEqual(props["operation"]["enum"], ["add", "subtract", "multiply", "divide"])
        self.assertEqual(props["a"]["type"], "number")
        self.assertEqual(props["b"]["type"], "number")

    def test_calculator_description_explicitly_documents_interface_and_examples(self):
        """Regression: Description explicitly documents operation, a, b, forbids expression, and gives examples."""
        desc = self.calculator.description

        # Required fields documented
        self.assertIn("operation", desc)
        self.assertIn("a", desc)
        self.assertIn("b", desc)

        # Allowed operations documented
        for op in ["add", "subtract", "multiply", "divide"]:
            self.assertIn(op, desc)

        # Explicitly forbids 'expression'
        self.assertIn("expression", desc)
        self.assertIn('{"expression":"25 * 4"}', desc)
        self.assertIn("MUST NOT use an 'expression' field", desc)

        # Concrete examples included
        self.assertIn('{"operation":"add","a":10,"b":5}', desc)
        self.assertIn('{"operation":"subtract","a":10,"b":5}', desc)
        self.assertIn('{"operation":"multiply","a":25,"b":4}', desc)
        self.assertIn('{"operation":"divide","a":20,"b":5}', desc)

    def test_model_facing_generated_tool_schema_structure(self):
        """Regression: generate_tool_schema produces clean model-facing schema without expression."""
        model_schema = generate_tool_schema(self.calculator)

        self.assertEqual(model_schema["tool"], "calculator")
        self.assertEqual(model_schema["allowed_operations"], ["add", "subtract", "multiply", "divide"])
        self.assertEqual(sorted(model_schema["required"]), ["a", "b", "operation"])

        params = model_schema["parameters"]
        self.assertIn("operation", params)
        self.assertIn("a", params)
        self.assertIn("b", params)
        self.assertNotIn("expression", params)

        self.assertEqual(params["operation"]["allowed_values"], ["add", "subtract", "multiply", "divide"])
        self.assertIn("expression", model_schema["safety_boundary"])

    def test_calculator_multiplication_25_times_4_exact_interface(self):
        """Regression: multiply 25 x 4 -> 100 using structured interface."""
        result = self.calculator.execute({
            "operation": "multiply",
            "a": 25,
            "b": 4,
        })
        self.assertEqual(result, 100)

    def test_calculator_rejects_expression_argument(self):
        """Regression: CalculatorTool rejects {'expression': '25 * 4'} without executing or evaluating."""
        with self.assertRaises(InvalidArgumentError) as ctx:
            self.calculator.execute({"expression": "25 * 4"})
        self.assertIn("Missing required argument", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
