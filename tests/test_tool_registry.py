import unittest

from core.interfaces.tool_registry import (
    DuplicateToolError,
    ToolNotFoundError,
    ToolRegistry,
)
from tools.calculator import CalculatorTool


class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.calculator = CalculatorTool()

    def test_register_tool(self):
        self.registry.register(self.calculator)
        self.assertIn("calculator", self.registry.list_tools())

    def test_retrieve_tool(self):
        self.registry.register(self.calculator)
        retrieved = self.registry.get("calculator")
        self.assertIs(retrieved, self.calculator)

    def test_list_registered_tools(self):
        self.assertEqual(self.registry.list_tools(), [])
        self.registry.register(self.calculator)
        self.assertEqual(self.registry.list_tools(), ["calculator"])

    def test_get_tool_metadata(self):
        self.registry.register(self.calculator)
        metadata_list = self.registry.get_tool_metadata()
        self.assertEqual(len(metadata_list), 1)
        self.assertEqual(metadata_list[0]["name"], "calculator")

    def test_execute_registered_tool(self):
        self.registry.register(self.calculator)
        result = self.registry.execute("calculator", {"operation": "add", "a": 2, "b": 3})
        self.assertEqual(result, 5)

        # Also test with kwargs
        result_kw = self.registry.execute("calculator", operation="multiply", a=3, b=4)
        self.assertEqual(result_kw, 12)

    def test_duplicate_registration_rejected(self):
        self.registry.register(self.calculator)
        with self.assertRaises(DuplicateToolError):
            self.registry.register(self.calculator)
        with self.assertRaises(ValueError):
            self.registry.register(self.calculator)

    def test_unknown_tool_lookup_rejected(self):
        with self.assertRaises(ToolNotFoundError):
            self.registry.get("nonexistent_tool")
        with self.assertRaises(KeyError):
            self.registry.get("nonexistent_tool")

    def test_unknown_tool_execution_rejected(self):
        with self.assertRaises(ToolNotFoundError):
            self.registry.execute("nonexistent_tool", {"operation": "add", "a": 1, "b": 1})
        with self.assertRaises(KeyError):
            self.registry.execute("nonexistent_tool", {"operation": "add", "a": 1, "b": 1})

    def test_register_invalid_tool_rejected(self):
        with self.assertRaises(TypeError):
            self.registry.register("not_a_tool")  # type: ignore


if __name__ == "__main__":
    unittest.main()
