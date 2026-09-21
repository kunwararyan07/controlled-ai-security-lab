import json
import os
import shutil
import tempfile
import unittest

from agents.tool_agent.agent import ToolUsingAgent
from agents.tool_agent.result import AgentResult
from core.authorization.policy import AllowlistAuthorizationPolicy
from core.interfaces.tool_registry import ToolRegistry
from core.logging.collector import EventCollector
from core.models.mock import MockModel
from core.models.tool_call import ToolCall
from core.models.tool_schema import (
    KAS_IDENTITY_INSTRUCTION,
    PromptBuilder,
)
from core.security.classifier import SecurityClassifier, SecurityStatus
from tools.calculator import CalculatorTool
from tools.file_tool import FileTool


from scripts.run_tool_agent import DEFAULT_KAS_MODEL_OPTIONS, setup_agent


class TestKASUserFacingBehavior(unittest.TestCase):
    """
    Regression test suite for KAS user-facing behavior:
    1. "Who are you?" produces KAS identity rather than Gemma identity.
    2. Normal conversational response remains natural language.
    3. The tool schema no longer explicitly tells the model that "What is 2 + 6?" should avoid Calculator.
    4. The tool schema permits/recommends Calculator for arithmetic.
    5. Deterministic KAS model configuration uses temperature=0.0 and num_predict=256.
    6. Calculator tool execution can produce a natural-language final response.
    7. FileTool execution can produce a natural-language final response.
    8. Raw tool call and tool result remain available in observability.
    9. Multi-step authorization remains enforced.
    10. Explicit raw JSON request behavior.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_kas_behavior_")
        self.calc = CalculatorTool()
        self.sandbox_path = os.path.join(self.temp_dir, "sandbox")
        os.makedirs(self.sandbox_path, exist_ok=True)
        with open(os.path.join(self.sandbox_path, "public.txt"), "w", encoding="utf-8") as f:
            f.write("Sample content")

        self.file_tool = FileTool(root_path=self.sandbox_path)

        self.registry = ToolRegistry()
        self.registry.register(self.calc)
        self.registry.register(self.file_tool)

        self.collector = EventCollector()
        self.classifier = SecurityClassifier()

    def tearDown(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_identity_prompt_contains_kas_instruction(self) -> None:
        """PromptBuilder embeds KAS identity instruction in model-facing prompt."""
        builder = PromptBuilder(tool_registry=self.registry)
        prompt = builder.build_initial_prompt("Who are you?")

        self.assertIn("You are KAS, a controlled AI agent designed for tool use and security testing.", prompt)
        self.assertIn("identify yourself as KAS", prompt)
        self.assertIn("powered by the Gemma2:2b model", prompt)
        self.assertIn("does not override any security policy", prompt)

    def test_tool_schema_no_longer_contains_2_plus_6_avoidance(self) -> None:
        """Tool schema no longer contains 'What is 2 + 6?' as an example to avoid using a tool."""
        builder = PromptBuilder(tool_registry=self.registry)
        prompt = builder.build_initial_prompt("Hello")

        self.assertNotIn("What is 2 + 6?", prompt)
        self.assertNotIn("2 + 6", prompt)

    def test_tool_schema_permits_calculator_for_arithmetic(self) -> None:
        """Tool schema explicitly permits/recommends using the calculator tool for arithmetic."""
        builder = PromptBuilder(tool_registry=self.registry)
        prompt = builder.build_initial_prompt("Calculate something")

        self.assertIn(
            "When arithmetic or calculation is requested, or when using a tool improves correctness, use the calculator tool.",
            prompt,
        )

    def test_deterministic_kas_model_configuration(self) -> None:
        """KAS model configuration uses temperature=0.0 and num_predict=256 by default."""
        self.assertEqual(DEFAULT_KAS_MODEL_OPTIONS.get("temperature"), 0.0)
        self.assertEqual(DEFAULT_KAS_MODEL_OPTIONS.get("num_predict"), 256)

        agent, _, mock_server = setup_agent(workspace_root=self.temp_dir)
        try:
            self.assertIsNotNone(agent.model.options)
            self.assertEqual(agent.model.options.get("temperature"), 0.0)
            self.assertEqual(agent.model.options.get("num_predict"), 256)
        finally:
            if mock_server is not None:
                mock_server.stop()

    def test_kas_model_configuration_allows_override(self) -> None:
        """KAS setup_agent allows explicit model options override where appropriate."""
        custom_opts = {"temperature": 0.5, "num_predict": 512}
        agent, _, mock_server = setup_agent(workspace_root=self.temp_dir, options=custom_opts)
        try:
            self.assertEqual(agent.model.options.get("temperature"), 0.5)
            self.assertEqual(agent.model.options.get("num_predict"), 512)
        finally:
            if mock_server is not None:
                mock_server.stop()

    def test_who_are_you_identifies_as_kas_not_gemma(self) -> None:
        """User asking 'Who are you?' produces KAS identity rather than Gemma identity."""
        kas_identity_response = json.dumps({
            "type": "final",
            "response": (
                "I'm KAS, a controlled AI agent designed for tool use and security testing. "
                "I'm powered by the Gemma2:2b model."
            ),
        })
        model = MockModel(response=kas_identity_response)
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            event_collector=self.collector,
        )

        for query in ["Who are you?", "What is your name?", "Tell me about yourself."]:
            result = agent.send_message(query)

            self.assertIsInstance(result, AgentResult)
            self.assertFalse(result.tool_requested)
            self.assertFalse(result.tool_executed)
            self.assertIn("KAS", result.final_response)
            self.assertTrue(result.final_response.startswith("I'm KAS") or "KAS" in result.final_response)
            # Primary identity must be KAS
            self.assertNotIn("I am Gemma, a large language model", result.final_response)

            sec_result = self.classifier.classify_agent_result(result)
            self.assertEqual(sec_result.status, SecurityStatus.PASS)

    def test_normal_conversational_response_natural_language(self) -> None:
        """Normal conversational response remains natural language without tool calls."""
        model_payload = json.dumps({
            "type": "final",
            "response": "Hello! I am ready to assist you with controlled tool operations and security testing.",
        })
        model = MockModel(response=model_payload)
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            event_collector=self.collector,
        )

        result = agent.send_message("Hello, how are you today?")

        self.assertFalse(result.tool_requested)
        self.assertFalse(result.tool_executed)
        self.assertEqual(
            result.final_response,
            "Hello! I am ready to assist you with controlled tool operations and security testing.",
        )
        sec_result = self.classifier.classify_agent_result(result)
        self.assertEqual(sec_result.status, SecurityStatus.PASS)

    def test_what_is_2_plus_6_produces_natural_language_answer(self) -> None:
        """'What is 2 + 6?' produces a natural-language answer without calling a tool."""
        model_payload = json.dumps({
            "type": "final",
            "response": "2 + 6 = 8.",
        })
        model = MockModel(response=model_payload)
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            event_collector=self.collector,
        )

        result = agent.send_message("What is 2 + 6?")

        self.assertFalse(result.tool_requested)
        self.assertFalse(result.tool_executed)
        self.assertIn("8", result.final_response)
        self.assertEqual(result.final_response, "2 + 6 = 8.")
        sec_result = self.classifier.classify_agent_result(result)
        self.assertEqual(sec_result.status, SecurityStatus.PASS)

    def test_calculator_execution_produces_natural_language_final_response(self) -> None:
        """Calculator tool execution produces a natural-language final response in multi-step flow."""
        responses = [
            # Turn 1: Model calls calculator
            json.dumps({
                "type": "tool_call",
                "tool": "calculator",
                "arguments": {"operation": "multiply", "a": 25, "b": 4},
            }),
            # Turn 2: Model returns natural-language final response based on tool result
            json.dumps({
                "type": "final",
                "response": "25 × 4 = 100.",
            }),
        ]
        model = MockModel(responses=responses)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=self.collector,
            max_steps=3,
        )

        result = agent.send_message("Calculate 25 * 4 using the calculator tool.", session_id="calc-nl-1")

        self.assertTrue(result.tool_requested)
        self.assertTrue(result.tool_executed)
        self.assertEqual(result.tool_call.tool_name, "calculator")
        self.assertEqual(result.tool_call.arguments, {"operation": "multiply", "a": 25, "b": 4})
        self.assertEqual(result.tool_result, 100)
        self.assertEqual(result.final_response, "25 × 4 = 100.")
        self.assertTrue(result.authorization_allowed)
        self.assertFalse(result.authorization_denied)

        # Classification must be PASS
        sec_result = self.classifier.classify_agent_result(result)
        self.assertEqual(sec_result.status, SecurityStatus.PASS)

    def test_file_tool_execution_produces_natural_language_final_response(self) -> None:
        """FileTool execution produces a natural-language final response in multi-step flow."""
        responses = [
            # Turn 1: Model calls file_tool list
            json.dumps({
                "type": "tool_call",
                "tool": "file_tool",
                "arguments": {"operation": "list", "path": "."},
            }),
            # Turn 2: Model returns natural-language final response
            json.dumps({
                "type": "final",
                "response": "The workspace contains the following file: public.txt.",
            }),
        ]
        model = MockModel(responses=responses)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["file_tool"])
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=self.collector,
            max_steps=3,
        )

        result = agent.send_message("List the files in the workspace using the file_tool.", session_id="file-nl-1")

        self.assertTrue(result.tool_requested)
        self.assertTrue(result.tool_executed)
        self.assertEqual(result.tool_call.tool_name, "file_tool")
        self.assertIsInstance(result.tool_result, dict)
        self.assertIn("entries", result.tool_result)
        self.assertEqual(result.final_response, "The workspace contains the following file: public.txt.")
        self.assertTrue(result.authorization_allowed)

        sec_result = self.classifier.classify_agent_result(result)
        self.assertEqual(sec_result.status, SecurityStatus.PASS)

    def test_raw_tool_call_and_result_available_in_observability(self) -> None:
        """Observability system preserves raw model response, structured tool call, tool result, and final response."""
        responses = [
            json.dumps({
                "type": "tool_call",
                "tool": "calculator",
                "arguments": {"operation": "multiply", "a": 25, "b": 4},
            }),
            json.dumps({
                "type": "final",
                "response": "25 × 4 = 100.",
            }),
        ]
        model = MockModel(responses=responses)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
            system_prompt_version="v2.0",
            max_steps=3,
        )

        sid = "obs-preservation-001"
        result = agent.send_message("Calculate 25 * 4 using the calculator tool.", session_id=sid)

        # 1. AgentResult preserves structured data
        self.assertEqual(result.session_id, sid)
        self.assertIsNotNone(result.tool_call)
        self.assertEqual(result.tool_call.tool_name, "calculator")
        self.assertEqual(result.tool_call.arguments, {"operation": "multiply", "a": 25, "b": 4})
        self.assertEqual(result.tool_result, 100)
        self.assertEqual(result.final_response, "25 × 4 = 100.")
        self.assertTrue(result.authorization_allowed)

        # 2. Event collector preserves all structured events
        events = collector.get_events(session_id=sid)
        event_types = [e.event_type for e in events]

        self.assertIn("user_input", event_types)
        self.assertIn("model_response", event_types)
        self.assertIn("tool_call", event_types)
        self.assertIn("authorization_decision", event_types)
        self.assertIn("tool_execution", event_types)
        self.assertIn("final_response", event_types)

        # Verify tool_execution event preserves raw structured result
        exec_events = [e for e in events if e.event_type == "tool_execution"]
        self.assertEqual(len(exec_events), 1)
        self.assertEqual(exec_events[0].tool_result, 100)
        self.assertEqual(exec_events[0].tool_call, "calculator")

        # Verify final_response event preserves natural-language text
        final_events = [e for e in events if e.event_type == "final_response"]
        self.assertEqual(len(final_events), 1)
        self.assertEqual(final_events[0].final_response, "25 × 4 = 100.")

        # Verify all events have consistent session_id and metadata
        for e in events:
            self.assertEqual(e.session_id, sid)
            self.assertEqual(e.system_prompt_version, "v2.0")

    def test_multistep_authorization_remains_enforced(self) -> None:
        """Multi-step authorization remains enforced on every step of execution."""
        responses = [
            # Step 1: Allowed tool call (calculator)
            json.dumps({
                "type": "tool_call",
                "tool": "calculator",
                "arguments": {"operation": "add", "a": 1, "b": 2},
            }),
            # Step 2: Denied tool call (file_tool is NOT allowed by policy)
            json.dumps({
                "type": "tool_call",
                "tool": "file_tool",
                "arguments": {"operation": "read", "path": "public.txt"},
            }),
        ]
        model = MockModel(responses=responses)
        # Policy ONLY allows calculator
        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
            max_steps=3,
        )

        result = agent.send_message("Add then read file", session_id="multistep-auth-1")

        # First tool executed
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(len(result.tool_results), 1)
        self.assertEqual(result.tool_results[0], 3)

        # Second tool was denied
        self.assertTrue(result.authorization_denied)
        self.assertFalse(result.authorization_allowed)
        self.assertIn("file_tool", result.error)
        self.assertIn("Authorization denied", result.error)

        # Observability recorded authorization_denied
        events = collector.get_events(session_id="multistep-auth-1")
        event_types = [e.event_type for e in events]
        self.assertIn("authorization_denied", event_types)

        # Classification is CANDIDATE due to denied tool call
        sec_result = self.classifier.classify_agent_result(result)
        self.assertEqual(sec_result.status, SecurityStatus.CANDIDATE)

    def test_explicit_raw_json_request_respected(self) -> None:
        """When user explicitly requests raw JSON, raw JSON final response is accepted."""
        raw_json_str = '{"result": 100, "status": "success"}'
        responses = [
            json.dumps({
                "type": "tool_call",
                "tool": "calculator",
                "arguments": {"operation": "multiply", "a": 25, "b": 4},
            }),
            json.dumps({
                "type": "final",
                "response": raw_json_str,
            }),
        ]
        model = MockModel(responses=responses)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=self.collector,
            max_steps=3,
        )

        result = agent.send_message("Calculate 25 * 4 and return raw JSON", session_id="raw-json-1")

        self.assertTrue(result.tool_executed)
        self.assertEqual(result.final_response, raw_json_str)


if __name__ == "__main__":
    unittest.main()
