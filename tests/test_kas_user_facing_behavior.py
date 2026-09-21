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


if __name__ == "__main__":
    unittest.main()
