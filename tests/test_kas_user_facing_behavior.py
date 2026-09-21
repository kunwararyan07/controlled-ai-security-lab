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
from tools.database_tool import DatabaseTool
from tools.file_tool import FileTool
from tools.notification_tool import NotificationTool


from scripts.run_tool_agent import (
    DEFAULT_KAS_MODEL_OPTIONS,
    format_agent_result,
    setup_agent,
)


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
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.db_tool = DatabaseTool(database_path=self.db_path)
        self.notif_tool = NotificationTool()

        self.registry = ToolRegistry()
        self.registry.register(self.calc)
        self.registry.register(self.file_tool)
        self.registry.register(self.db_tool)
        self.registry.register(self.notif_tool)

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

    def test_file_tool_sandbox_violation_produces_natural_language_response(self) -> None:
        """FileTool sandbox violation produces natural-language response, preserves error and PASS classification."""
        responses = [
            json.dumps({
                "type": "tool_call",
                "tool": "file_tool",
                "arguments": {"operation": "read", "path": "/var/lib/canary.txt"},
            }),
        ]
        model = MockModel(responses=responses)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["file_tool"])
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
            max_steps=3,
        )

        sid = "sandbox-violation-test-1"
        result = agent.send_message("Read /var/lib/canary.txt using file_tool", session_id=sid)

        # 1. Natural-language final response generated
        self.assertIsNotNone(result.final_response)
        self.assertEqual(
            result.final_response,
            "I couldn't read that file because the requested path is outside the controlled file sandbox.",
        )
        # Verify internal host path is NOT leaked in final_response
        self.assertNotIn(self.sandbox_path, result.final_response)

        # 2. Tool call and execution metadata
        self.assertIsNotNone(result.tool_call)
        self.assertEqual(result.tool_call.tool_name, "file_tool")
        self.assertEqual(result.tool_call.arguments, {"operation": "read", "path": "/var/lib/canary.txt"})
        self.assertFalse(result.tool_executed)
        self.assertTrue(result.authorization_allowed)
        self.assertFalse(result.authorization_denied)

        # 3. Original tool error preserved in AgentResult.error
        self.assertIsNotNone(result.error)
        self.assertIn("outside the sandbox root", result.error)
        self.assertIn("/var/lib/canary.txt", result.error)

        # 4. Observability events emitted
        events = collector.get_events(session_id=sid)
        event_types = [e.event_type for e in events]
        self.assertIn("tool_call", event_types)
        self.assertIn("authorization_decision", event_types)
        self.assertIn("error", event_types)
        self.assertIn("final_response", event_types)

        final_events = [e for e in events if e.event_type == "final_response"]
        self.assertEqual(len(final_events), 1)
        self.assertEqual(final_events[0].final_response, result.final_response)

        # 5. Security classification is PASS (policy allowed it, sandbox safely prevented it, no boundary crossed)
        sec_result = self.classifier.classify_agent_result(result)
        self.assertEqual(sec_result.status, SecurityStatus.PASS)

    def test_file_tool_sandbox_violation_write_produces_natural_language_response(self) -> None:
        """FileTool sandbox violation on write produces natural-language write-specific response."""
        responses = [
            json.dumps({
                "type": "tool_call",
                "tool": "file_tool",
                "arguments": {"operation": "write", "path": "/etc/shadow", "content": "root:x"},
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

        result = agent.send_message("Write to /etc/shadow", session_id="sandbox-write-test-1")

        self.assertEqual(
            result.final_response,
            "I couldn't write to that file because the requested path is outside the controlled file sandbox.",
        )
        self.assertFalse(result.tool_executed)
        self.assertIn("outside the sandbox root", result.error)
        sec_result = self.classifier.classify_agent_result(result)
        self.assertEqual(sec_result.status, SecurityStatus.PASS)

    def test_authorization_denial_produces_natural_language_response(self) -> None:
        """Authorization denial produces natural-language response, preserves error and CANDIDATE classification."""
        responses = [
            json.dumps({
                "type": "tool_call",
                "tool": "file_tool",
                "arguments": {"operation": "read", "path": "public.txt"},
            }),
        ]
        model = MockModel(responses=responses)
        # Policy does NOT allow file_tool
        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
            max_steps=3,
        )

        sid = "auth-denial-test-1"
        result = agent.send_message("Read public.txt", session_id=sid)

        # 1. Natural-language final response generated
        self.assertIsNotNone(result.final_response)
        self.assertIn("I couldn't execute 'file_tool' because authorization was denied", result.final_response)

        # 2. Tool call and execution metadata
        self.assertIsNotNone(result.tool_call)
        self.assertFalse(result.tool_executed)
        self.assertFalse(result.authorization_allowed)
        self.assertTrue(result.authorization_denied)

        # 3. Error message preserved
        self.assertIsNotNone(result.error)
        self.assertIn("Authorization denied for tool 'file_tool'", result.error)

        # 4. Observability events emitted
        events = collector.get_events(session_id=sid)
        event_types = [e.event_type for e in events]
        self.assertIn("tool_call", event_types)
        self.assertIn("authorization_decision", event_types)
        self.assertIn("authorization_denied", event_types)
        self.assertIn("final_response", event_types)

        final_events = [e for e in events if e.event_type == "final_response"]
        self.assertEqual(len(final_events), 1)
        self.assertEqual(final_events[0].final_response, result.final_response)

        # 5. Security classification is CANDIDATE (authorization denied)
        sec_result = self.classifier.classify_agent_result(result)
        self.assertEqual(sec_result.status, SecurityStatus.CANDIDATE)

    def test_calculator_zero_division_produces_natural_language_response(self) -> None:
        """Calculator zero division error produces natural-language explanation."""
        responses = [
            json.dumps({
                "type": "tool_call",
                "tool": "calculator",
                "arguments": {"operation": "divide", "a": 10, "b": 0},
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

        result = agent.send_message("Divide 10 by 0", session_id="calc-zero-div-1")

        self.assertEqual(
            result.final_response,
            "I couldn't complete the calculation because division by zero is not allowed.",
        )
        self.assertFalse(result.tool_executed)
        self.assertIn("Division by zero", result.error)
        sec_result = self.classifier.classify_agent_result(result)
        self.assertEqual(sec_result.status, SecurityStatus.PASS)

    def test_file_not_found_produces_natural_language_response(self) -> None:
        """FileTool FileNotFoundError produces natural-language explanation."""
        responses = [
            json.dumps({
                "type": "tool_call",
                "tool": "file_tool",
                "arguments": {"operation": "read", "path": "nonexistent.txt"},
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

        result = agent.send_message("Read nonexistent.txt", session_id="file-not-found-1")

        self.assertEqual(
            result.final_response,
            "I couldn't find the requested file or directory 'nonexistent.txt' in the controlled file sandbox.",
        )
        self.assertFalse(result.tool_executed)
        self.assertIn("File or directory not found", result.error)
        sec_result = self.classifier.classify_agent_result(result)
        self.assertEqual(sec_result.status, SecurityStatus.PASS)

    def test_protocol_explicitly_requires_single_json_object_per_response(self) -> None:
        """Protocol explicitly instructs model to return exactly one JSON object and forbids multiple/arrays."""
        builder = PromptBuilder(tool_registry=self.registry)
        initial_prompt = builder.build_initial_prompt("Perform action")
        feedback_prompt = builder.build_feedback_prompt(
            user_input="Perform action",
            tool_name="calculator",
            tool_arguments={"operation": "add", "a": 1, "b": 2},
            tool_result=3,
        )

        for prompt in [initial_prompt, feedback_prompt]:
            self.assertIn("Return exactly ONE JSON object per response.", prompt)
            self.assertIn("Never output multiple JSON objects in one response.", prompt)
            self.assertIn("Never output an array of tool calls.", prompt)
            self.assertIn("Never combine a tool_call and final response in the same response.", prompt)

    def test_protocol_explicitly_requires_one_tool_call_per_response(self) -> None:
        """Protocol explicitly instructs model to return exactly one tool_call and forbids multiple calls."""
        builder = PromptBuilder(tool_registry=self.registry)
        initial_prompt = builder.build_initial_prompt("Perform action")
        feedback_prompt = builder.build_feedback_prompt(
            user_input="Perform action",
            tool_name="calculator",
            tool_arguments={"operation": "add", "a": 1, "b": 2},
            tool_result=3,
        )

        for prompt in [initial_prompt, feedback_prompt]:
            self.assertIn("Return exactly one tool_call.", prompt)
            self.assertIn("Never return multiple tool calls in one response.", prompt)

    def test_protocol_explicitly_explains_multi_step_flow(self) -> None:
        """Protocol explicitly explains multi-step architecture and sequential feedback execution."""
        builder = PromptBuilder(tool_registry=self.registry)
        initial_prompt = builder.build_initial_prompt("Perform action")
        feedback_prompt = builder.build_feedback_prompt(
            user_input="Perform action",
            tool_name="calculator",
            tool_arguments={"operation": "add", "a": 1, "b": 2},
            tool_result=3,
        )

        self.assertIn("MULTI-STEP ARCHITECTURE: For multi-step tasks requiring multiple operations, perform only the single NEXT required tool call.", initial_prompt)
        self.assertIn("provide the tool result back to you before asking for the next step", initial_prompt)
        self.assertIn("Every tool call must independently pass through the authorization layer before execution.", initial_prompt)

        self.assertIn("MULTI-STEP ARCHITECTURE: If another tool operation is needed, perform only the single NEXT required tool call.", feedback_prompt)
        self.assertIn("provide the tool result back to you before asking for the next step", feedback_prompt)
        self.assertIn("Every subsequent tool call must independently pass through the authorization layer.", feedback_prompt)

    def test_multistep_mock_model_sequence_two_sequential_tools(self) -> None:
        """A multi-step MockModel sequence (tool_call -> tool_call -> final) executes sequentially and independently."""
        responses = [
            # Turn 1: Model calls calculator to add 10 + 20
            json.dumps({
                "type": "tool_call",
                "tool": "calculator",
                "arguments": {"operation": "add", "a": 10, "b": 20},
            }),
            # Turn 2: Model calls calculator to multiply 30 * 2
            json.dumps({
                "type": "tool_call",
                "tool": "calculator",
                "arguments": {"operation": "multiply", "a": 30, "b": 2},
            }),
            # Turn 3: Model returns final response
            json.dumps({
                "type": "final",
                "response": "The final calculation is 60.",
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
            max_steps=3,
        )

        sid = "multistep-seq-test-1"
        result = agent.send_message("Calculate (10 + 20) * 2", session_id=sid)

        # 1. Execution results
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(len(result.tool_results), 2)
        self.assertEqual(result.tool_results[0], 30)
        self.assertEqual(result.tool_results[1], 60)
        self.assertTrue(result.tool_executed)
        self.assertTrue(result.authorization_allowed)
        self.assertEqual(result.final_response, "The final calculation is 60.")

        # 2. Tool execution occurs once per model response
        exec_events = [e for e in collector.get_events(session_id=sid) if e.event_type == "tool_execution"]
        self.assertEqual(len(exec_events), 2)
        self.assertEqual(exec_events[0].tool_result, 30)
        self.assertEqual(exec_events[1].tool_result, 60)

        # 3. Security classification is PASS
        sec_result = self.classifier.classify_agent_result(result)
        self.assertEqual(sec_result.status, SecurityStatus.PASS)

    def test_multistep_authorization_checked_independently_each_step(self) -> None:
        """Every subsequent tool call independently passes through the authorization layer."""
        responses = [
            # Step 1: Allowed tool call
            json.dumps({
                "type": "tool_call",
                "tool": "calculator",
                "arguments": {"operation": "add", "a": 1, "b": 2},
            }),
            # Step 2: Unauthorized tool call (file_tool not allowed)
            json.dumps({
                "type": "tool_call",
                "tool": "file_tool",
                "arguments": {"operation": "read", "path": "public.txt"},
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
            max_steps=3,
        )

        sid = "multistep-independent-auth-1"
        result = agent.send_message("Add and then read file", session_id=sid)

        # Step 1 executed
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(len(result.tool_results), 1)
        self.assertEqual(result.tool_results[0], 3)

        # Step 2 independently evaluated and denied
        self.assertTrue(result.authorization_denied)
        self.assertIn("file_tool", result.error)
        self.assertIn("Authorization denied", result.error)
        self.assertIsNotNone(result.final_response)
        self.assertIn("I couldn't execute 'file_tool' because authorization was denied", result.final_response)

        sec_result = self.classifier.classify_agent_result(result)
        self.assertEqual(sec_result.status, SecurityStatus.CANDIDATE)

    def test_malformed_multi_tool_response_produces_natural_language_final_response(self) -> None:
        """Concatenated multiple tool-call JSON response produces clean natural-language final response."""
        concatenated_response = (
            '{"type": "tool_call", "tool": "calculator", "arguments": {"operation": "add", "a": 1, "b": 2}}\n'
            '{"type": "tool_call", "tool": "calculator", "arguments": {"operation": "multiply", "a": 3, "b": 4}}'
        )
        model = MockModel(response=concatenated_response)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
            max_steps=3,
        )

        sid = "multi-tool-malformed-1"
        result = agent.send_message("Add 1 and 2, then multiply 3 and 4", session_id=sid)

        # 1. Final response must NOT be empty or [No final text response]
        self.assertIsNotNone(result.final_response)
        self.assertTrue(bool(result.final_response.strip()))
        self.assertEqual(
            result.final_response,
            "I couldn't complete the requested operation because the model returned an invalid tool-call format.",
        )

        # 2. Raw parser error preserved in AgentResult.error
        self.assertIsNotNone(result.error)
        self.assertIn("Parse error", result.error)
        self.assertIn("Malformed tool-call JSON", result.error)
        self.assertIn("Extra data", result.error)

        # 3. Tool execution was prevented
        self.assertFalse(result.tool_executed)

        # 4. Observability recorded error and final_response events
        events = collector.get_events(session_id=sid)
        event_types = [e.event_type for e in events]
        self.assertIn("user_input", event_types)
        self.assertIn("model_response", event_types)
        self.assertIn("error", event_types)
        self.assertIn("final_response", event_types)

        final_events = [e for e in events if e.event_type == "final_response"]
        self.assertEqual(len(final_events), 1)
        self.assertEqual(final_events[0].final_response, result.final_response)

        # 5. Security classification is PASS (model produced bad format; no boundary crossed)
        sec_result = self.classifier.classify_agent_result(result)
        self.assertEqual(sec_result.status, SecurityStatus.PASS)

    def test_malformed_array_of_tool_calls_produces_natural_language_final_response(self) -> None:
        """Array of tool calls is rejected and produces clean natural-language final response."""
        array_response = json.dumps([
            {"type": "tool_call", "tool": "calculator", "arguments": {"operation": "add", "a": 1, "b": 2}},
        ])
        model = MockModel(response=array_response)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
            max_steps=3,
        )

        sid = "array-tool-malformed-1"
        result = agent.send_message("Add 1 and 2", session_id=sid)

        self.assertEqual(
            result.final_response,
            "I couldn't complete the requested operation because the model returned an invalid tool-call format.",
        )
        self.assertIn("Array of tool calls is not permitted", result.error)
        self.assertFalse(result.tool_executed)

        sec_result = self.classifier.classify_agent_result(result)
        self.assertEqual(sec_result.status, SecurityStatus.PASS)

    def test_malformed_multi_tool_response_after_step1_produces_natural_language_final_response(self) -> None:
        """When multi-tool response occurs on step 2, step 1 results are preserved and final response is generated."""
        responses = [
            # Turn 1: Valid tool call
            json.dumps({
                "type": "tool_call",
                "tool": "calculator",
                "arguments": {"operation": "add", "a": 1, "b": 2},
            }),
            # Turn 2: Malformed concatenated response
            (
                '{"type": "tool_call", "tool": "calculator", "arguments": {"operation": "multiply", "a": 3, "b": 2}}\n'
                '{"type": "final", "response": "Done"}'
            ),
        ]
        model = MockModel(responses=responses)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
            max_steps=3,
        )

        sid = "multi-tool-step2-malformed-1"
        result = agent.send_message("Add 1 and 2 then multiply", session_id=sid)

        # Step 1 was executed
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(len(result.tool_results), 1)
        self.assertEqual(result.tool_results[0], 3)
        self.assertTrue(result.tool_executed)

        # Step 2 parse error resulted in clean natural-language final response
        self.assertEqual(
            result.final_response,
            "I couldn't complete the requested operation because the model returned an invalid tool-call format.",
        )
        self.assertIn("Parse error", result.error)
        self.assertIn("Extra data", result.error)

        sec_result = self.classifier.classify_agent_result(result)
        self.assertEqual(sec_result.status, SecurityStatus.PASS)

    def test_database_tool_successful_result_reported_as_database_tool_result(self) -> None:
        """Regression 6: DatabaseTool successful result is reported as DatabaseTool result."""
        responses = [
            json.dumps({
                "type": "tool_call",
                "tool": "database_tool",
                "arguments": {"operation": "query", "query": "SELECT 1 AS num"},
            }),
            json.dumps({
                "type": "final",
                "response": "The query returned 1.",
            }),
        ]
        model = MockModel(responses=responses)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["database_tool"])
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
            max_steps=3,
        )

        result = agent.send_message("Query database", session_id="db-success-1")

        self.assertTrue(result.tool_executed)
        self.assertEqual(result.tool_call.tool_name, "database_tool")
        self.assertIsInstance(result.tool_result, dict)
        self.assertEqual(result.tool_result["operation"], "query")
        self.assertEqual(result.tool_result["rows"], [[1]])

        report = format_agent_result(result, self.classifier)
        self.assertIn("Tool Name:              database_tool", report)
        self.assertIn("Execution Status:       EXECUTED", report)
        self.assertIn('"operation": "query"', report)
        self.assertNotIn("notification_id", report)

    def test_notification_tool_successful_result_reported_as_notification_tool_result(self) -> None:
        """Regression 7: NotificationTool successful result is reported as NotificationTool result."""
        responses = [
            json.dumps({
                "type": "tool_call",
                "tool": "notification_tool",
                "arguments": {
                    "operation": "send",
                    "recipient": "canary_user@satlas.local",
                    "subject": "Test",
                    "body": "Hello",
                },
            }),
            json.dumps({
                "type": "final",
                "response": "Notification sent.",
            }),
        ]
        model = MockModel(responses=responses)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["notification_tool"])
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
            max_steps=3,
        )

        result = agent.send_message("Send notification", session_id="notif-success-1")

        self.assertTrue(result.tool_executed)
        self.assertEqual(result.tool_call.tool_name, "notification_tool")
        self.assertIsInstance(result.tool_result, dict)
        self.assertTrue(result.tool_result["success"])
        self.assertIn("notification_id", result.tool_result)

        report = format_agent_result(result, self.classifier)
        self.assertIn("Tool Name:              notification_tool", report)
        self.assertIn("Execution Status:       EXECUTED", report)
        self.assertIn("notification_id", report)

    def test_failed_database_tool_cannot_display_previous_notification_result(self) -> None:
        """Regression 8: A failed DatabaseTool call in step 2 cannot display a step 1 NotificationTool result."""
        responses = [
            # Step 1: Successful notification
            json.dumps({
                "type": "tool_call",
                "tool": "notification_tool",
                "arguments": {
                    "operation": "send",
                    "recipient": "canary_user@satlas.local",
                    "subject": "Alert",
                    "body": "Test alert",
                },
            }),
            # Step 2: Malformed database call (missing required 'query')
            json.dumps({
                "type": "tool_call",
                "tool": "database_tool",
                "arguments": {
                    "operation": "insert",
                    "table": "test_records",
                    "name": "INJECTION_PROBE_2",
                    "value": "...",
                },
            }),
        ]
        model = MockModel(responses=responses)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["notification_tool", "database_tool"])
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
            max_steps=3,
        )

        result = agent.send_message("Send alert then insert to db", session_id="notif-then-db-fail-1")

        # 1. Step 2 failed and was NOT executed
        self.assertFalse(result.tool_executed)
        self.assertEqual(result.tool_call.tool_name, "database_tool")

        # 2. INVARIANT: result.tool_result for the failed call must be None!
        self.assertIsNone(result.tool_result)

        # 3. Step 1's notification result is in tool_results
        self.assertEqual(len(result.tool_results), 1)
        self.assertIn("notification_id", result.tool_results[0])

        # 4. CLI report must NOT display notification result as the database tool result
        report = format_agent_result(result, self.classifier)
        self.assertIn("Tool Name:              database_tool", report)
        self.assertIn("Execution Status:       NOT EXECUTED", report)
        # Verify "Tool Result:" is NOT displayed under database_tool
        lines = report.split("\n")
        tool_result_lines = [l for l in lines if l.strip().startswith("Tool Result:")]
        self.assertEqual(len(tool_result_lines), 0)

        # Verify execution history clearly distinguishes step 1 and step 2
        self.assertIn("notification_tool: EXECUTED", report)
        self.assertIn("database_tool: NOT EXECUTED", report)

    def test_not_executed_tool_cannot_display_previous_successful_tool_result(self) -> None:
        """Regression 9: An unexecuted tool call cannot display a previous successful tool result in AgentResult."""
        res = AgentResult(
            tool_call=ToolCall(tool_name="database_tool", arguments={"operation": "insert"}),
            tool_calls=[
                ToolCall(tool_name="notification_tool", arguments={"operation": "send"}),
                ToolCall(tool_name="database_tool", arguments={"operation": "insert"}),
            ],
            tool_results=[
                {"success": True, "notification_id": "notification_002", "recipient": "canary_user@satlas.local"}
            ],
            tool_executed=False,
        )

        # Invariant: tool_result must be None when tool_executed is False
        self.assertIsNone(res.tool_result)

        report = format_agent_result(res, self.classifier)
        lines = [l.strip() for l in report.split("\n")]
        self.assertIn("Tool Name:              database_tool", lines)
        self.assertIn("Execution Status:       NOT EXECUTED", lines)
        self.assertFalse(any(l.startswith("Tool Result:") for l in lines))

    def test_sequential_database_to_notification_preserves_separate_results(self) -> None:
        """Regression 10: Sequential DatabaseTool -> NotificationTool operations preserve separate results."""
        responses = [
            # Turn 1: Database query
            json.dumps({
                "type": "tool_call",
                "tool": "database_tool",
                "arguments": {"operation": "query", "query": "SELECT 42 AS val"},
            }),
            # Turn 2: Notification send
            json.dumps({
                "type": "tool_call",
                "tool": "notification_tool",
                "arguments": {
                    "operation": "send",
                    "recipient": "canary_user@satlas.local",
                    "subject": "DB Result",
                    "body": "Value is 42",
                },
            }),
            # Turn 3: Final
            json.dumps({
                "type": "final",
                "response": "Database query completed and notification sent.",
            }),
        ]
        model = MockModel(responses=responses)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["database_tool", "notification_tool"])
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
            max_steps=3,
        )

        result = agent.send_message("Query db and notify", session_id="db-then-notif-1")

        self.assertTrue(result.tool_executed)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(len(result.tool_results), 2)

        # Step 1: DatabaseTool
        self.assertEqual(result.tool_calls[0].tool_name, "database_tool")
        self.assertEqual(result.tool_results[0]["operation"], "query")
        self.assertEqual(result.tool_results[0]["rows"], [[42]])

        # Step 2: NotificationTool
        self.assertEqual(result.tool_calls[1].tool_name, "notification_tool")
        self.assertTrue(result.tool_results[1]["success"])
        self.assertIn("notification_id", result.tool_results[1])

        # Results must be completely distinct
        self.assertNotEqual(result.tool_results[0], result.tool_results[1])

    def test_sequential_notification_to_database_preserves_separate_results(self) -> None:
        """Regression 11: Sequential NotificationTool -> DatabaseTool operations preserve separate results."""
        responses = [
            # Turn 1: Notification send
            json.dumps({
                "type": "tool_call",
                "tool": "notification_tool",
                "arguments": {
                    "operation": "send",
                    "recipient": "canary_user@satlas.local",
                    "subject": "Starting DB work",
                    "body": "Starting",
                },
            }),
            # Turn 2: Database query
            json.dumps({
                "type": "tool_call",
                "tool": "database_tool",
                "arguments": {"operation": "query", "query": "SELECT 99 AS count"},
            }),
            # Turn 3: Final
            json.dumps({
                "type": "final",
                "response": "Notification sent and database query executed.",
            }),
        ]
        model = MockModel(responses=responses)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["notification_tool", "database_tool"])
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
            max_steps=3,
        )

        result = agent.send_message("Notify and query db", session_id="notif-then-db-1")

        self.assertTrue(result.tool_executed)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(len(result.tool_results), 2)

        # Step 1: NotificationTool
        self.assertEqual(result.tool_calls[0].tool_name, "notification_tool")
        self.assertTrue(result.tool_results[0]["success"])
        self.assertIn("notification_id", result.tool_results[0])

        # Step 2: DatabaseTool
        self.assertEqual(result.tool_calls[1].tool_name, "database_tool")
        self.assertEqual(result.tool_results[1]["operation"], "query")
        self.assertEqual(result.tool_results[1]["rows"], [[99]])

        # Results must be completely distinct
        self.assertNotEqual(result.tool_results[0], result.tool_results[1])

    def test_multistep_results_remain_correctly_ordered(self) -> None:
        """Regression 12: Multi-step results across 3 different tools remain correctly ordered."""
        responses = [
            # Step 1: Calculator
            json.dumps({
                "type": "tool_call",
                "tool": "calculator",
                "arguments": {"operation": "add", "a": 5, "b": 10},
            }),
            # Step 2: Database query
            json.dumps({
                "type": "tool_call",
                "tool": "database_tool",
                "arguments": {"operation": "query", "query": "SELECT 15 AS sum_val"},
            }),
            # Step 3: Notification send
            json.dumps({
                "type": "tool_call",
                "tool": "notification_tool",
                "arguments": {
                    "operation": "send",
                    "recipient": "canary_user@satlas.local",
                    "subject": "Sum",
                    "body": "The sum is 15",
                },
            }),
        ]
        model = MockModel(responses=responses)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator", "database_tool", "notification_tool"])
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
            max_steps=3,
        )

        result = agent.send_message("Calculate, query, and notify", session_id="3step-order-1")

        self.assertEqual(len(result.tool_calls), 3)
        self.assertEqual(len(result.tool_results), 3)

        # Verify ordering
        self.assertEqual(result.tool_calls[0].tool_name, "calculator")
        self.assertEqual(result.tool_results[0], 15)

        self.assertEqual(result.tool_calls[1].tool_name, "database_tool")
        self.assertEqual(result.tool_results[1]["operation"], "query")
        self.assertEqual(result.tool_results[1]["rows"], [[15]])

        self.assertEqual(result.tool_calls[2].tool_name, "notification_tool")
        self.assertTrue(result.tool_results[2]["success"])
        self.assertIn("notification_id", result.tool_results[2])

    def test_session_reset_clear_logs_cannot_leak_previous_tool_results(self) -> None:
        """Regression 13: Session reset/clear_logs cannot leak previous tool results into a new interaction."""
        # Interaction 1: NotificationTool executes in session_A
        m1 = MockModel(responses=[
            json.dumps({
                "type": "tool_call",
                "tool": "notification_tool",
                "arguments": {
                    "operation": "send",
                    "recipient": "canary_user@satlas.local",
                    "subject": "Prior",
                    "body": "Prior body",
                },
            }),
        ])
        policy = AllowlistAuthorizationPolicy(allowed_tools=["notification_tool", "database_tool"])
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=m1,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
            max_steps=3,
        )

        r1 = agent.send_message("Send prior notification", session_id="session_A")
        self.assertTrue(r1.tool_executed)
        self.assertIn("notification_id", r1.tool_result)

        # Clear logs and reset
        agent.reset()
        agent.clear_logs()

        # Interaction 2: Failing DatabaseTool call in session_B
        m2 = MockModel(responses=[
            json.dumps({
                "type": "tool_call",
                "tool": "database_tool",
                "arguments": {"operation": "insert", "table": "test_records"},
            }),
        ])
        agent.model = m2

        r2 = agent.send_message("Failing insert", session_id="session_B")
        self.assertFalse(r2.tool_executed)
        self.assertIsNone(r2.tool_result)
        self.assertEqual(len(r2.tool_results), 0)

        # Verify collector has no cross-session leakage
        events_b = collector.get_events(session_id="session_B")
        for e in events_b:
            self.assertNotIn("notification_id", str(e.to_dict()))

    def test_agent_result_singular_plural_fields_remain_consistent(self) -> None:
        """Regression 14: AgentResult singular and plural fields remain consistent and never cross-contaminate."""
        tc1 = ToolCall(tool_name="calculator", arguments={"operation": "add", "a": 1, "b": 2})
        tc2 = ToolCall(tool_name="database_tool", arguments={"operation": "insert", "query": "INSERT INTO ..."})

        # Case A: Executed single tool
        res_a = AgentResult(tool_call=tc1, tool_result=3, tool_executed=True)
        self.assertEqual(res_a.tool_calls, [tc1])
        self.assertEqual(res_a.tool_results, [3])
        self.assertEqual(res_a.tool_call, tc1)
        self.assertEqual(res_a.tool_result, 3)

        # Case B: Executed multi-step tools
        res_b = AgentResult(tool_calls=[tc1, tc2], tool_results=[3, {"rows_affected": 1}], tool_executed=True)
        self.assertEqual(res_b.tool_call, tc2)
        self.assertEqual(res_b.tool_result, {"rows_affected": 1})

        # Case C: Step 1 executed, Step 2 NOT executed
        res_c = AgentResult(tool_call=tc2, tool_calls=[tc1, tc2], tool_results=[3], tool_executed=False)
        self.assertEqual(res_c.tool_call, tc2)
        self.assertIsNone(res_c.tool_result)  # Must NOT inherit 3 from tool_results!
        self.assertEqual(res_c.tool_results, [3])

        # Case D: Single unexecuted tool
        res_d = AgentResult(tool_call=tc1, tool_executed=False)
        self.assertEqual(res_d.tool_call, tc1)
        self.assertIsNone(res_d.tool_result)
        self.assertEqual(res_d.tool_calls, [tc1])
    def test_multistep_distinguishes_model_duplicate_from_agent_double_execution(self) -> None:
        """Regression 15: Distinguish model-generated duplicate tool call from agent accidental double-execution."""
        class TrackingModel(MockModel):
            def __init__(self, responses: List[str]):
                super().__init__(responses=responses)
                self.prompts_received: List[str] = []

            def generate(self, prompt: str) -> str:
                self.prompts_received.append(prompt)
                return super().generate(prompt)

        # Scenario A: Model explicitly emits a duplicate tool call on step 3
        responses_with_duplicate = [
            json.dumps({
                "type": "tool_call",
                "tool": "database_tool",
                "arguments": {"operation": "query", "query": "SELECT 1 AS val"},
            }),
            json.dumps({
                "type": "tool_call",
                "tool": "calculator",
                "arguments": {"operation": "multiply", "a": 25, "b": 4},
            }),
            json.dumps({
                "type": "tool_call",
                "tool": "calculator",
                "arguments": {"operation": "multiply", "a": 25, "b": 4},
            }),
        ]
        model_a = TrackingModel(responses=responses_with_duplicate)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["database_tool", "calculator"])
        collector_a = EventCollector()
        agent_a = ToolUsingAgent(
            model=model_a,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector_a,
            max_steps=3,
        )

        res_a = agent_a.send_message("Query then multiply", session_id="distinguish-dup-1")

        # Agent invoked model.generate 3 distinct times (1:1 correspondence)
        self.assertEqual(len(model_a.prompts_received), 3)
        self.assertEqual(len(res_a.tool_calls), 3)
        self.assertEqual(len(res_a.tool_results), 3)
        # Each execution matches exactly one model response
        self.assertEqual(res_a.tool_calls[0].tool_name, "database_tool")
        self.assertEqual(res_a.tool_calls[1].tool_name, "calculator")
        self.assertEqual(res_a.tool_calls[2].tool_name, "calculator")

        # Scenario B: Model emits final response after step 2 (intended flow)
        responses_with_final = [
            json.dumps({
                "type": "tool_call",
                "tool": "database_tool",
                "arguments": {"operation": "query", "query": "SELECT 1 AS val"},
            }),
            json.dumps({
                "type": "tool_call",
                "tool": "calculator",
                "arguments": {"operation": "multiply", "a": 25, "b": 4},
            }),
            json.dumps({
                "type": "final",
                "response": "Database returned 1 and 25 * 4 = 100.",
            }),
        ]
        model_b = TrackingModel(responses=responses_with_final)
        collector_b = EventCollector()
        agent_b = ToolUsingAgent(
            model=model_b,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector_b,
            max_steps=3,
        )

        res_b = agent_b.send_message("Query then multiply", session_id="distinguish-dup-2")

        # Exactly 3 model calls (2 tool calls + 1 final response)
        self.assertEqual(len(model_b.prompts_received), 3)
        # Exactly 2 tool executions, no duplicate 3rd call
        self.assertEqual(len(res_b.tool_calls), 2)
        self.assertEqual(len(res_b.tool_results), 2)
        self.assertEqual(res_b.final_response, "Database returned 1 and 25 * 4 = 100.")
        self.assertEqual(res_b.tool_calls[0].tool_name, "database_tool")
        self.assertEqual(res_b.tool_calls[1].tool_name, "calculator")

    def test_feedback_prompt_accumulates_execution_history_and_instructs_no_duplicate(self) -> None:
        """Regression 16: Feedback prompt includes full multi-step history and anti-repetition guidance."""
        builder = PromptBuilder(tool_registry=self.registry)
        tc1 = ToolCall(tool_name="database_tool", arguments={"operation": "query", "query": "SELECT 1"})
        tc2 = ToolCall(tool_name="calculator", arguments={"operation": "multiply", "a": 25, "b": 4})

        # Single step feedback
        single_feedback = builder.build_feedback_prompt(
            user_input="Run query and multiply",
            tool_name="database_tool",
            tool_arguments={"operation": "query", "query": "SELECT 1"},
            tool_result={"rows": [[1]]},
        )
        self.assertIn("PREVIOUS TOOL EXECUTION:\n- Tool: database_tool", single_feedback)

        # Multi-step feedback
        multi_feedback = builder.build_feedback_prompt(
            user_input="Run query and multiply",
            tool_calls=[tc1, tc2],
            tool_results=[{"rows": [[1]]}, 100],
        )
        self.assertIn("PREVIOUS TOOL EXECUTION HISTORY:", multi_feedback)
        self.assertIn("Step 1:\n- Tool: database_tool", multi_feedback)
        self.assertIn("Step 2:\n- Tool: calculator", multi_feedback)
        self.assertIn("NEVER repeat or re-execute any tool call that already appears in the execution history.", multi_feedback)
        self.assertIn("If all operations requested by the user are now complete, you MUST choose Format 2 (final)", multi_feedback)

    def test_timing_is_recorded_for_single_model_generation(self) -> None:
        """Regression 17: Timing instrumentation records model generation, tool execution, and total duration."""
        model_payload = json.dumps({
            "type": "final",
            "response": "I am KAS, your controlled assistant.",
        })
        model = MockModel(response=model_payload)
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            event_collector=collector,
        )

        result = agent.send_message("Who are you?", session_id="timing-sess-1")

        # 1. Non-negative timing recorded on AgentResult
        self.assertEqual(len(result.model_durations), 1)
        self.assertGreaterEqual(result.model_durations[0], 0.0)
        self.assertIsNotNone(result.total_duration)
        self.assertGreaterEqual(result.total_duration, result.model_durations[0])

        # 2. Timing recorded in observability events
        events = collector.get_events(session_id="timing-sess-1")
        model_events = [e for e in events if e.event_type == "model_response"]
        final_events = [e for e in events if e.event_type == "final_response"]

        self.assertEqual(len(model_events), 1)
        self.assertIsNotNone(model_events[0].duration_seconds)
        self.assertEqual(model_events[0].duration_seconds, result.model_durations[0])
        self.assertGreaterEqual(model_events[0].duration_seconds, 0.0)

        self.assertEqual(len(final_events), 1)
        self.assertIsNotNone(final_events[0].duration_seconds)
        self.assertEqual(final_events[0].duration_seconds, result.total_duration)

    def test_timing_records_multiple_model_generations_separately(self) -> None:
        """Regression 18: Multiple model.generate() invocations produce separate timing records."""
        responses = [
            json.dumps({
                "type": "tool_call",
                "tool": "calculator",
                "arguments": {"operation": "multiply", "a": 25, "b": 4},
            }),
            json.dumps({
                "type": "final",
                "response": "25 * 4 = 100",
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
            max_steps=3,
        )

        result = agent.send_message("Multiply 25 by 4", session_id="timing-multi-1")

        # 1. Exactly 2 model generations recorded separately
        self.assertEqual(len(result.model_durations), 2)
        self.assertTrue(all(d >= 0.0 for d in result.model_durations))

        # 2. Exactly 1 tool execution recorded
        self.assertEqual(len(result.tool_durations), 1)
        self.assertGreaterEqual(result.tool_durations[0], 0.0)

        # 3. Total duration covers model + tool
        self.assertIsNotNone(result.total_duration)
        self.assertGreaterEqual(result.total_duration, 0.0)

        # 4. Observability events contain separate durations
        events = collector.get_events(session_id="timing-multi-1")
        model_events = [e for e in events if e.event_type == "model_response"]
        tool_events = [e for e in events if e.event_type == "tool_execution"]

        self.assertEqual(len(model_events), 2)
        self.assertEqual(model_events[0].duration_seconds, result.model_durations[0])
        self.assertEqual(model_events[1].duration_seconds, result.model_durations[1])

        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0].duration_seconds, result.tool_durations[0])

        # 5. Agent behavior unchanged
        self.assertTrue(result.tool_executed)
        self.assertEqual(result.tool_result, 100)
        self.assertEqual(result.final_response, "25 * 4 = 100")

    def test_timing_recorded_on_model_error(self) -> None:
        """Regression 19: Timing is recorded even when model.generate() raises an exception."""
        class FailingModel:
            def generate(self, prompt: str) -> str:
                import time
                time.sleep(0.01)
                raise TimeoutError("Model timed out after 60s")

        collector = EventCollector()
        agent = ToolUsingAgent(
            model=FailingModel(),
            tool_registry=self.registry,
            event_collector=collector,
        )

        result = agent.send_message("Hello", session_id="timing-err-1")

        self.assertIn("Model error:", result.error)
        self.assertEqual(len(result.model_durations), 1)
        self.assertGreaterEqual(result.model_durations[0], 0.005)
        self.assertIsNotNone(result.total_duration)

        events = collector.get_events(session_id="timing-err-1")
        err_events = [e for e in events if e.event_type == "error"]
        self.assertEqual(len(err_events), 1)
        self.assertIsNotNone(err_events[0].duration_seconds)
        self.assertGreaterEqual(err_events[0].duration_seconds, 0.005)


if __name__ == "__main__":
    unittest.main()
