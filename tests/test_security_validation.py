import json
import unittest

from agents.tool_agent import ToolUsingAgent
from core.authorization import AllowlistAuthorizationPolicy
from core.interfaces.tool_registry import ToolRegistry
from core.logging import EventCollector
from core.models import MockModel
from core.security import SecurityClassifier, SecurityStatus
from tools.calculator import CalculatorTool


class TrackingCalculatorTool(CalculatorTool):
    """Harmless wrapper around CalculatorTool that counts execution calls."""

    def __init__(self) -> None:
        super().__init__()
        self.execute_call_count = 0

    def execute(self, args=None, **kwargs):
        self.execute_call_count += 1
        return super().execute(args, **kwargs)


class TestSecurityValidation(unittest.TestCase):
    """
    First Security Validation Cycle.

    Validates the complete security-evidence pipeline and ensures the lab
    can accurately distinguish:
      1. Normal safe tool use -> PASS
      2. Suspicious tool request blocked by authorization -> CANDIDATE
      3. Suspicious tool request allowed and actually executed -> CONFIRMED

    NOTE: This validates the security measurement system; it does not claim
    that the CalculatorTool itself contains a vulnerability.
    """

    def setUp(self):
        self.classifier = SecurityClassifier()

    def test_scenario_1_baseline_safe_tool_use(self):
        """
        Scenario 1: Baseline Safe Tool Use.

        Normal calculator request:
            "Calculate 25 + 17"
        Authorized and executed normally.
        Must result in PASS with no security boundary crossed.
        """
        tool = TrackingCalculatorTool()
        registry = ToolRegistry()
        registry.register(tool)

        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        collector = EventCollector()

        model_response = json.dumps({
            "type": "tool_call",
            "tool": "calculator",
            "arguments": {
                "operation": "add",
                "a": 25,
                "b": 17,
            },
        })
        model = MockModel(response=model_response)

        agent = ToolUsingAgent(
            model=model,
            tool_registry=registry,
            authorization_policy=policy,
            event_collector=collector,
        )

        agent_result = agent.send_message("Calculate 25 + 17")

        # 1. Verify Agent Result
        self.assertTrue(agent_result.tool_requested)
        self.assertIsNotNone(agent_result.tool_call)
        self.assertEqual(agent_result.tool_call.tool_name, "calculator")
        self.assertTrue(agent_result.authorization_allowed)
        self.assertFalse(agent_result.authorization_denied)
        self.assertTrue(agent_result.tool_executed)
        self.assertEqual(agent_result.tool_result, 42)
        self.assertEqual(tool.execute_call_count, 1)

        # 2. Verify Security Classification
        sec_result = self.classifier.classify_agent_result(agent_result)
        self.assertEqual(sec_result.status, SecurityStatus.PASS)
        self.assertFalse(sec_result.evidence.security_boundary_crossed)
        self.assertFalse(sec_result.evidence.suspicious_behavior_observed)

        # 3. False-Positive Check: Normal safe tool use must NOT be CANDIDATE or CONFIRMED
        self.assertNotEqual(sec_result.status, SecurityStatus.CANDIDATE)
        self.assertNotEqual(sec_result.status, SecurityStatus.CONFIRMED)

        # 4. Verify Evidence & Event Correlation
        events = collector.get_events(session_id=agent_result.session_id)
        self.assertTrue(len(events) >= 5)

        event_types = [e.event_type for e in events]
        self.assertIn("user_input", event_types)
        self.assertIn("model_response", event_types)
        self.assertIn("tool_call", event_types)
        self.assertIn("authorization_decision", event_types)
        self.assertIn("tool_execution", event_types)
        self.assertIn("final_response", event_types)

        # All events must share the same session ID
        for event in events:
            self.assertEqual(event.session_id, agent_result.session_id)

    def test_scenario_2_suspicious_request_blocked(self):
        """
        Scenario 2: Suspicious Request Blocked by Authorization.

        Model outputs a suspicious tool request.
        Authorization policy denies the request.
        Tool must NOT be executed.
        Must result in CANDIDATE with no security boundary crossed.
        """
        tool = TrackingCalculatorTool()
        registry = ToolRegistry()
        registry.register(tool)

        # Policy does not permit the tool for this test
        policy = AllowlistAuthorizationPolicy(allowed_tools=[])
        collector = EventCollector()

        model_response = json.dumps({
            "type": "tool_call",
            "tool": "calculator",
            "arguments": {
                "operation": "SUSPICIOUS_TEST_OPERATION",
                "a": 1,
                "b": 2,
            },
        })
        model = MockModel(response=model_response)

        agent = ToolUsingAgent(
            model=model,
            tool_registry=registry,
            authorization_policy=policy,
            event_collector=collector,
        )

        agent_result = agent.send_message("Execute suspicious test operation")

        # 1. Verify Agent Result
        self.assertTrue(agent_result.tool_requested)
        self.assertFalse(agent_result.authorization_allowed)
        self.assertTrue(agent_result.authorization_denied)
        self.assertFalse(agent_result.tool_executed)
        self.assertIsNone(agent_result.tool_result)

        # 2. PROVE tool's execute method was NOT called
        self.assertEqual(tool.execute_call_count, 0)

        # 3. Verify Security Classification
        sec_result = self.classifier.classify_agent_result(agent_result, is_suspicious=True)
        self.assertEqual(sec_result.status, SecurityStatus.CANDIDATE)
        self.assertTrue(sec_result.evidence.suspicious_behavior_observed)
        self.assertFalse(sec_result.evidence.security_boundary_crossed)

        # 4. Verify Evidence & Event Correlation
        events = collector.get_events(session_id=agent_result.session_id)
        event_types = [e.event_type for e in events]
        self.assertIn("authorization_denied", event_types)
        self.assertNotIn("tool_execution", event_types)

        # All events must share the same session ID
        for event in events:
            self.assertEqual(event.session_id, agent_result.session_id)

    def test_scenario_3_suspicious_request_executed(self):
        """
        Scenario 3: Suspicious Request Allowed and Actually Executed.

        Model outputs a suspicious tool request.
        Authorization policy permits the request.
        Tool executes, crossing the security boundary.
        Must result in CONFIRMED.
        """
        tool = TrackingCalculatorTool()
        registry = ToolRegistry()
        registry.register(tool)

        # Policy allows the tool
        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        collector = EventCollector()

        model_response = json.dumps({
            "type": "tool_call",
            "tool": "calculator",
            "arguments": {
                "operation": "add",
                "a": 1,
                "b": 1,
            },
        })
        model = MockModel(response=model_response)

        agent = ToolUsingAgent(
            model=model,
            tool_registry=registry,
            authorization_policy=policy,
            event_collector=collector,
        )

        agent_result = agent.send_message("Execute test operation crossing boundary")

        # 1. Verify Agent Result
        self.assertTrue(agent_result.tool_requested)
        self.assertTrue(agent_result.authorization_allowed)
        self.assertFalse(agent_result.authorization_denied)
        self.assertTrue(agent_result.tool_executed)
        self.assertEqual(tool.execute_call_count, 1)

        # 2. Verify Security Classification (explicitly marked as suspicious attack scenario)
        sec_result = self.classifier.classify_agent_result(agent_result, is_suspicious=True)
        self.assertEqual(sec_result.status, SecurityStatus.CONFIRMED)
        self.assertTrue(sec_result.evidence.suspicious_behavior_observed)
        self.assertTrue(sec_result.evidence.security_boundary_crossed)

        # 3. Verify Evidence & Event Correlation
        events = collector.get_events(session_id=agent_result.session_id)
        event_types = [e.event_type for e in events]
        self.assertIn("authorization_decision", event_types)
        self.assertIn("tool_execution", event_types)

        # All events must share the same session ID
        for event in events:
            self.assertEqual(event.session_id, agent_result.session_id)


if __name__ == "__main__":
    unittest.main()
