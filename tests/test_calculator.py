import unittest

from core.interfaces.tool import Tool
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


if __name__ == "__main__":
    unittest.main()
