import unittest
from typing import Any, Dict, Optional

from core.authorization import (
    AllowlistAuthorizationPolicy,
    AuthorizationDecision,
    AuthorizationDeniedError,
    AuthorizationPolicy,
    DefaultToolAuthorizationPolicy,
)
from core.interfaces.tool import Tool
from core.interfaces.tool_registry import ToolNotFoundError, ToolRegistry
from tools.calculator import CalculatorTool


class MockTrackingTool(Tool):
    """Tool that tracks whether its execute method was called."""

    def __init__(self, tool_name: str = "mock_tracking") -> None:
        self._name = tool_name
        self.call_count = 0
        self.last_args: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Mock tool for tracking execution calls."

    def execute(self, args: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        self.call_count += 1
        self.last_args = args
        return "executed"


class ArgumentInspectingPolicy(AuthorizationPolicy):
    """Policy that inspects arguments and denies division operations."""

    def evaluate(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthorizationDecision:
        if arguments and arguments.get("operation") == "divide":
            return AuthorizationDecision(
                allowed=False,
                reason="Division operations are restricted by policy.",
            )
        return AuthorizationDecision(
            allowed=True,
            reason="Operation is permitted.",
        )


class TestAuthorization(unittest.TestCase):
    def test_authorization_decision_attributes_and_conversion(self):
        decision_allowed = AuthorizationDecision(allowed=True, reason="Permitted")
        self.assertTrue(decision_allowed.allowed)
        self.assertEqual(decision_allowed.reason, "Permitted")
        self.assertTrue(bool(decision_allowed))
        self.assertEqual(
            decision_allowed.to_dict(),
            {"allowed": True, "reason": "Permitted"},
        )

        decision_denied = AuthorizationDecision(allowed=False, reason="Denied by rule")
        self.assertFalse(decision_denied.allowed)
        self.assertEqual(decision_denied.reason, "Denied by rule")
        self.assertFalse(bool(decision_denied))
        self.assertEqual(
            decision_denied.to_dict(),
            {"allowed": False, "reason": "Denied by rule"},
        )

    def test_allowlist_policy_allows_permitted_tool(self):
        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        decision = policy.evaluate("calculator")
        self.assertTrue(decision.allowed)
        self.assertIn("permitted", decision.reason.lower())

    def test_allowlist_policy_denies_unpermitted_tool(self):
        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        decision = policy.evaluate("shell")
        self.assertFalse(decision.allowed)
        self.assertIn("not in the allowed tools list", decision.reason.lower())

    def test_allowlist_policy_dynamic_modification(self):
        policy = AllowlistAuthorizationPolicy()
        self.assertFalse(policy.evaluate("calculator").allowed)

        policy.add_tool("calculator")
        self.assertTrue(policy.evaluate("calculator").allowed)

        policy.remove_tool("calculator")
        self.assertFalse(policy.evaluate("calculator").allowed)

    def test_default_tool_authorization_policy_alias(self):
        self.assertIs(DefaultToolAuthorizationPolicy, AllowlistAuthorizationPolicy)

    def test_argument_validation_hook_in_policy(self):
        policy = ArgumentInspectingPolicy()

        # Add operation should be allowed
        decision_add = policy.evaluate("calculator", {"operation": "add", "a": 1, "b": 2})
        self.assertTrue(decision_add.allowed)

        # Divide operation should be denied
        decision_div = policy.evaluate("calculator", {"operation": "divide", "a": 1, "b": 2})
        self.assertFalse(decision_div.allowed)
        self.assertIn("restricted", decision_div.reason.lower())

    def test_registry_execution_authorized(self):
        registry = ToolRegistry()
        calculator = CalculatorTool()
        registry.register(calculator)

        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        result = registry.execute(
            "calculator",
            {"operation": "add", "a": 10, "b": 5},
            authorization_policy=policy,
        )
        self.assertEqual(result, 15)

    def test_registry_execution_denied_raises_authorization_denied_error(self):
        registry = ToolRegistry()
        calculator = CalculatorTool()
        registry.register(calculator)

        policy = AllowlistAuthorizationPolicy(allowed_tools=["other_tool"])

        with self.assertRaises(AuthorizationDeniedError) as cm:
            registry.execute(
                "calculator",
                {"operation": "add", "a": 10, "b": 5},
                authorization_policy=policy,
            )

        self.assertEqual(cm.exception.tool_name, "calculator")
        self.assertIn("not in the allowed tools list", cm.exception.reason.lower())
        self.assertIsInstance(cm.exception, PermissionError)

    def test_tool_not_executed_when_authorization_denied(self):
        registry = ToolRegistry()
        mock_tool = MockTrackingTool("sensitive_tool")
        registry.register(mock_tool)

        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])

        with self.assertRaises(AuthorizationDeniedError):
            registry.execute("sensitive_tool", {"action": "delete"}, authorization_policy=policy)

        # Ensure execute was never called on the tool
        self.assertEqual(mock_tool.call_count, 0)
        self.assertIsNone(mock_tool.last_args)

    def test_registry_default_policy(self):
        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        registry = ToolRegistry(authorization_policy=policy)
        calculator = CalculatorTool()
        mock_tool = MockTrackingTool("dangerous_tool")
        registry.register(calculator)
        registry.register(mock_tool)

        # Allowed by default policy
        result = registry.execute("calculator", {"operation": "multiply", "a": 3, "b": 4})
        self.assertEqual(result, 12)

        # Denied by default policy
        with self.assertRaises(AuthorizationDeniedError):
            registry.execute("dangerous_tool")
        self.assertEqual(mock_tool.call_count, 0)

    def test_call_level_policy_overrides_registry_default(self):
        default_policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        registry = ToolRegistry(authorization_policy=default_policy)
        calculator = CalculatorTool()
        registry.register(calculator)

        # Override with policy that denies calculator
        override_policy = AllowlistAuthorizationPolicy(allowed_tools=[])
        with self.assertRaises(AuthorizationDeniedError):
            registry.execute(
                "calculator",
                {"operation": "add", "a": 1, "b": 1},
                authorization_policy=override_policy,
            )

    def test_resolve_tool_before_authorization_check(self):
        # Even if a policy would deny, unregistered tool raises ToolNotFoundError
        policy = AllowlistAuthorizationPolicy(allowed_tools=[])
        registry = ToolRegistry()

        with self.assertRaises(ToolNotFoundError):
            registry.execute("nonexistent_tool", authorization_policy=policy)

    def test_backward_compatibility_without_policy(self):
        registry = ToolRegistry()
        calculator = CalculatorTool()
        registry.register(calculator)

        result = registry.execute("calculator", {"operation": "subtract", "a": 20, "b": 8})
        self.assertEqual(result, 12)


if __name__ == "__main__":
    unittest.main()
