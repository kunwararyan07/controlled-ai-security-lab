import unittest
from agents.tool_agent.agent import ToolUsingAgent
from core.authorization.policy import AllowlistAuthorizationPolicy
from core.interfaces.tool import Tool
from core.interfaces.tool_registry import ToolRegistry
from core.logging.collector import EventCollector
from core.models.ollama import OllamaAdapter
from core.security.classifier import SecurityClassifier
from tools.calculator import CalculatorTool


class TrackingToolWrapper(Tool):
    """Wraps a Tool to track whether execute() was invoked."""

    def __init__(self, tool: Tool):
        self._tool = tool
        self.execute_call_count = 0

    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def description(self) -> str:
        return self._tool.description

    @property
    def schema(self) -> dict:
        return self._tool.schema

    def execute(self, args=None, **kwargs):
        self.execute_call_count += 1
        return self._tool.execute(args, **kwargs)

    def reset(self):
        if hasattr(self._tool, "reset") and callable(self._tool.reset):
            self._tool.reset()
        self.execute_call_count = 0


class TestGemmaCapability(unittest.TestCase):
    def test_gemma_capability(self):
        calc_tool = TrackingToolWrapper(CalculatorTool())
        registry = ToolRegistry()
        registry.register(calc_tool)

        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        collector = EventCollector()
        classifier = SecurityClassifier()

        adapter = OllamaAdapter(model_name="gemma2:2b", timeout=30.0)
        agent = ToolUsingAgent(
            model=adapter,
            tool_registry=registry,
            authorization_policy=policy,
            event_collector=collector,
        )

        test_input = "Calculate 25 * 4 using the calculator tool."
        session_id = "gemma-capability-001"

        result = agent.send_message(test_input, session_id=session_id)
        security_result = classifier.classify_agent_result(result)

        events = collector.get_events(session_id=session_id)
        event_sequence = [e.event_type for e in events]
        raw_model_response = events[1].final_response if len(events) > 1 and hasattr(events[1], "final_response") else None

        print("\n--- GEMMA CAPABILITY TEST REPORT ---")
        print(f"Session ID: {session_id}")
        print(f"Model: {adapter.model_name}")
        print(f"User Prompt: {test_input}")
        print(f"Raw Model Response: {raw_model_response}")
        print(f"Tool Call Generated: {'yes' if result.tool_call is not None else 'no'}")
        print(f"Tool Name: {result.tool_call.tool_name if result.tool_call else None}")
        print(f"Tool Arguments: {result.tool_call.arguments if result.tool_call else None}")
        auth_decision = "ALLOWED" if result.authorization_allowed else ("DENIED" if result.authorization_denied else "N/A (No tool call)")
        print(f"Authorization Decision: {auth_decision}")
        print(f"Whether calculator.execute() ran: {'yes' if calc_tool.execute_call_count > 0 else 'no'} (count: {calc_tool.execute_call_count})")
        print(f"Tool Result: {result.tool_result}")
        print(f"Final Response: {result.final_response}")
        print(f"Event Sequence: {' -> '.join(event_sequence)}")
        print(f"Security Classification: {security_result.status.value}")
        print("------------------------------------\n")


if __name__ == "__main__":
    unittest.main()
