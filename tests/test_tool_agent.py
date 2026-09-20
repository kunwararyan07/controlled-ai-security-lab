import json
import unittest

from agents.tool_agent import AgentResult, ToolUsingAgent
from core.authorization import AllowlistAuthorizationPolicy
from core.interfaces.tool_registry import ToolRegistry
from core.logging import EventCollector
from core.models import MockModel, ToolCall
from tools.calculator import CalculatorTool


class TestToolUsingAgent(unittest.TestCase):
    def setUp(self):
        self.calculator = CalculatorTool()
        self.registry = ToolRegistry()
        self.registry.register(self.calculator)

    def test_tool_call_dataclass(self):
        tc = ToolCall(tool_name="calculator", arguments={"operation": "add", "a": 1, "b": 2})
        self.assertEqual(tc.tool_name, "calculator")
        self.assertEqual(tc.arguments, {"operation": "add", "a": 1, "b": 2})
        self.assertEqual(
            tc.to_dict(),
            {"tool_name": "calculator", "arguments": {"operation": "add", "a": 1, "b": 2}},
        )

    def test_agent_final_response_json(self):
        model_payload = json.dumps({"type": "final", "content": "The calculation is complete."})
        model = MockModel(response=model_payload)
        agent = ToolUsingAgent(model=model, tool_registry=self.registry)

        result = agent.send_message("What is 2 + 2?")

        self.assertIsInstance(result, AgentResult)
        self.assertEqual(result.final_response, "The calculation is complete.")
        self.assertFalse(result.tool_requested)
        self.assertIsNone(result.tool_call)
        self.assertFalse(result.tool_executed)
        self.assertFalse(result.authorization_denied)
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.session_id)

    def test_agent_final_response_plain_text(self):
        model = MockModel(response="Plain text response from model.")
        agent = ToolUsingAgent(model=model, tool_registry=self.registry)

        result = agent.send_message("Hello")

        self.assertEqual(result.final_response, "Plain text response from model.")
        self.assertFalse(result.tool_requested)
        self.assertFalse(result.tool_executed)
        self.assertIsNone(result.error)

    def test_agent_tool_call_authorized_and_executed(self):
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "calculator",
            "arguments": {"operation": "add", "a": 10, "b": 5},
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
        )

        result = agent.send_message("Add 10 and 5")

        self.assertTrue(result.tool_requested)
        self.assertIsNotNone(result.tool_call)
        self.assertEqual(result.tool_call.tool_name, "calculator")
        self.assertEqual(result.tool_call.arguments, {"operation": "add", "a": 10, "b": 5})
        self.assertTrue(result.tool_executed)
        self.assertEqual(result.tool_result, 15)
        self.assertTrue(result.authorization_allowed)
        self.assertFalse(result.authorization_denied)
        self.assertIsNone(result.error)

    def test_agent_tool_call_denied_by_authorization(self):
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "calculator",
            "arguments": {"operation": "add", "a": 10, "b": 5},
        })
        model = MockModel(response=model_payload)
        # Empty policy denies all tools
        policy = AllowlistAuthorizationPolicy(allowed_tools=[])
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
        )

        result = agent.send_message("Add 10 and 5")

        self.assertTrue(result.tool_requested)
        self.assertFalse(result.tool_executed)
        self.assertIsNone(result.tool_result)
        self.assertFalse(result.authorization_allowed)
        self.assertTrue(result.authorization_denied)
        self.assertIsNotNone(result.authorization_reason)
        self.assertIn("not in the allowed tools list", result.authorization_reason)
        self.assertIsNotNone(result.error)

    def test_agent_tool_call_unknown_tool(self):
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "nonexistent_tool",
            "arguments": {},
        })
        model = MockModel(response=model_payload)
        agent = ToolUsingAgent(model=model, tool_registry=self.registry)

        result = agent.send_message("Run something")

        self.assertTrue(result.tool_requested)
        self.assertFalse(result.tool_executed)
        self.assertIsNone(result.tool_result)
        self.assertIsNotNone(result.error)
        self.assertIn("not found", result.error)

    def test_agent_tool_execution_error_handled(self):
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "calculator",
            "arguments": {"operation": "divide", "a": 10, "b": 0},
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
        )

        result = agent.send_message("Divide 10 by 0")

        self.assertTrue(result.tool_requested)
        self.assertTrue(result.authorization_allowed)
        self.assertFalse(result.authorization_denied)
        self.assertFalse(result.tool_executed)
        self.assertIsNone(result.tool_result)
        self.assertIsNotNone(result.error)
        self.assertIn("Division by zero", result.error)

    def test_agent_malformed_tool_call_response(self):
        # Missing tool name
        model_payload = json.dumps({"type": "tool_call", "arguments": {}})
        model = MockModel(response=model_payload)
        agent = ToolUsingAgent(model=model, tool_registry=self.registry)

        result = agent.send_message("Do something")

        self.assertFalse(result.tool_executed)
        self.assertIsNotNone(result.error)
        self.assertIn("Parse error", result.error)

    def test_mock_model_sequential_responses(self):
        responses = ["first response", "second response"]
        model = MockModel(responses=responses)

        self.assertEqual(model.generate("prompt 1"), "first response")
        self.assertEqual(model.generate("prompt 2"), "second response")
        # Repeating after sequence exhausted returns last response
        self.assertEqual(model.generate("prompt 3"), "second response")

    def test_agent_result_to_dict(self):
        tc = ToolCall(tool_name="calc", arguments={"a": 1})
        result = AgentResult(
            session_id="sess_123",
            final_response="Done",
            tool_call=tc,
            tool_executed=True,
            tool_result=42,
        )
        data = result.to_dict()
        self.assertEqual(data["session_id"], "sess_123")
        self.assertEqual(data["final_response"], "Done")
        self.assertTrue(data["tool_requested"])
        self.assertEqual(data["tool_call"]["tool_name"], "calc")
        self.assertTrue(data["tool_executed"])
        self.assertEqual(data["tool_result"], 42)

    def test_agent_observability_with_event_collector(self):
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "calculator",
            "arguments": {"operation": "multiply", "a": 3, "b": 4},
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
            system_prompt_version="v1.0",
        )

        result = agent.send_message("Multiply 3 by 4", session_id="fixed_session")

        self.assertEqual(result.session_id, "fixed_session")
        self.assertEqual(result.tool_result, 12)

        events = collector.get_events(session_id="fixed_session")
        self.assertTrue(len(events) >= 5)

        event_types = [e.event_type for e in events]
        self.assertIn("user_input", event_types)
        self.assertIn("model_response", event_types)
        self.assertIn("tool_call", event_types)
        self.assertIn("authorization_decision", event_types)
        self.assertIn("tool_execution", event_types)
        self.assertIn("final_response", event_types)

        # Verify all events share the session_id and system_prompt_version
        for event in events:
            self.assertEqual(event.session_id, "fixed_session")
            self.assertEqual(event.system_prompt_version, "v1.0")

    def test_agent_observability_when_denied(self):
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "calculator",
            "arguments": {"operation": "add", "a": 1, "b": 1},
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=[])
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
        )

        result = agent.send_message("Add 1 and 1")

        self.assertTrue(result.authorization_denied)
        events = collector.get_events(session_id=result.session_id)
        event_types = [e.event_type for e in events]
        self.assertIn("authorization_decision", event_types)
        self.assertIn("authorization_denied", event_types)
        self.assertNotIn("tool_execution", event_types)


if __name__ == "__main__":
    unittest.main()
