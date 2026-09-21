import io
import json
import os
import shutil
import sys
import tempfile
from unittest.mock import patch
import unittest

from agents.tool_agent.agent import ToolUsingAgent
from agents.tool_agent.result import AgentResult
from core.authorization.policy import AllowlistAuthorizationPolicy
from core.interfaces.tool_registry import ToolRegistry
from core.logging.collector import EventCollector
from core.models.mock import MockModel
from core.models.tool_call import ToolCall
from core.security.classifier import SecurityClassifier
from scripts.run_tool_agent import (
    format_agent_result,
    format_banner,
    handle_command,
    parse_args,
    setup_agent,
)
from tools.calculator import CalculatorTool


class TestKASCLIRunner(unittest.TestCase):
    """Test suite for KAS interactive CLI runner components."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_kas_cli_")
        self.calc = CalculatorTool()
        self.registry = ToolRegistry()
        self.registry.register(self.calc)
        self.policy = AllowlistAuthorizationPolicy(allowed_tools=["calculator"])
        self.collector = EventCollector()
        self.classifier = SecurityClassifier()

    def tearDown(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parse_args_defaults(self) -> None:
        """1. Argument parser defaults to gemma2:2b and safe localhost endpoint."""
        args = parse_args([])
        self.assertEqual(args.model, "gemma2:2b")
        self.assertEqual(args.endpoint, "http://127.0.0.1:11434")
        self.assertEqual(args.timeout, 120.0)
        self.assertEqual(args.max_steps, 3)
        self.assertIsNone(args.session_id)

    def test_parse_args_custom(self) -> None:
        """2. Argument parser accepts custom model and options."""
        args = parse_args([
            "--model", "qwen3:4b",
            "--endpoint", "http://127.0.0.1:11435",
            "--timeout", "45.0",
            "--max-steps", "5",
            "--session-id", "test-sess-001",
        ])
        self.assertEqual(args.model, "qwen3:4b")
        self.assertEqual(args.endpoint, "http://127.0.0.1:11435")
        self.assertEqual(args.timeout, 45.0)
        self.assertEqual(args.max_steps, 5)
        self.assertEqual(args.session_id, "test-sess-001")

    def test_parse_args_timeout_override(self) -> None:
        """2b. Argument parser allows explicitly overriding timeout."""
        args_60 = parse_args(["--timeout", "60.0"])
        self.assertEqual(args_60.timeout, 60.0)

        args_180 = parse_args(["--timeout", "180"])
        self.assertEqual(args_180.timeout, 180.0)

    def test_parse_args_help_text_contains_default_120(self) -> None:
        """2c. CLI help text documents default: 120.0."""
        from io import StringIO
        captured = StringIO()
        with patch("sys.stdout", captured):
            with self.assertRaises(SystemExit):
                parse_args(["--help"])
        self.assertIn("default: 120.0", captured.getvalue())

    def test_banner_formatting(self) -> None:
        """3. Banner prominently features KAS TOOL_CALLING AGENT."""
        banner = format_banner()
        self.assertIn("KAS TOOL_CALLING AGENT", banner)
        self.assertIn("╔", banner)
        self.assertIn("╚", banner)

    def test_format_agent_result_final_response(self) -> None:
        """4. format_agent_result displays final response and security classification."""
        result = AgentResult(
            session_id="kas-test-1",
            final_response="The answer is 100.",
        )
        output = format_agent_result(result, self.classifier)
        self.assertIn("KAS > The answer is 100.", output)
        self.assertIn("Tool Requested:         NO", output)
        self.assertIn("Security Status:        PASS", output)

    def test_format_agent_result_tool_call_executed(self) -> None:
        """5. format_agent_result displays tool call, authorization, and execution."""
        result = AgentResult(
            session_id="kas-test-2",
            tool_call=ToolCall(tool_name="calculator", arguments={"operation": "multiply", "a": 25, "b": 4}),
            tool_calls=[ToolCall(tool_name="calculator", arguments={"operation": "multiply", "a": 25, "b": 4})],
            tool_results=[100],
            tool_executed=True,
            tool_result=100,
            authorization_allowed=True,
            final_response="100",
        )
        output = format_agent_result(result, self.classifier)
        self.assertIn("KAS > 100", output)
        self.assertIn("Tool Requested:         YES", output)
        self.assertIn("Tool Name:              calculator", output)
        self.assertIn('"operation": "multiply"', output)
        self.assertIn("Authorization Decision: ALLOWED", output)
        self.assertIn("Execution Status:       EXECUTED", output)
        self.assertIn("Tool Result:            100", output)

    def test_format_agent_result_tool_call_denied(self) -> None:
        """6. format_agent_result displays authorization denial."""
        result = AgentResult(
            session_id="kas-test-3",
            tool_call=ToolCall(tool_name="command_tool", arguments={"command": "ls"}),
            authorization_denied=True,
            authorization_reason="Denied by policy",
            tool_executed=False,
            error="Authorization denied: Denied by policy",
        )
        output = format_agent_result(result, self.classifier)
        self.assertIn("Tool Requested:         YES", output)
        self.assertIn("Tool Name:              command_tool", output)
        self.assertIn("Authorization Decision: DENIED (Denied by policy)", output)
        self.assertIn("Execution Status:       NOT EXECUTED", output)
        self.assertIn("Error:                  Authorization denied: Denied by policy", output)

    def test_handle_command_help(self) -> None:
        """7. /help command prints help message and returns True."""
        agent = ToolUsingAgent(model=MockModel(response=""), tool_registry=self.registry)
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cont = handle_command("/help", agent, "sess-1")
        self.assertTrue(cont)
        self.assertIn("Available KAS Commands:", captured.getvalue())
        self.assertIn("/state", captured.getvalue())
        self.assertIn("/logs", captured.getvalue())
        self.assertIn("/clear", captured.getvalue())
        self.assertIn("/reset", captured.getvalue())
        self.assertIn("/quit", captured.getvalue())

    def test_handle_command_state(self) -> None:
        """8. /state command prints agent state and returns True."""
        agent = ToolUsingAgent(model=MockModel(response=""), tool_registry=self.registry)
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cont = handle_command("/state", agent, "sess-1")
        self.assertTrue(cont)
        self.assertIn("AGENT STATE", captured.getvalue())
        self.assertIn('"calculator"', captured.getvalue())

    def test_handle_command_logs(self) -> None:
        """9. /logs command displays structured events for the session."""
        agent = ToolUsingAgent(
            model=MockModel(response="hello"),
            tool_registry=self.registry,
            event_collector=self.collector,
        )
        agent.send_message("test", session_id="log-sess")

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cont = handle_command("/logs", agent, "log-sess")
        self.assertTrue(cont)
        self.assertIn("SESSION LOGS (log-sess)", captured.getvalue())
        self.assertIn("user_input", captured.getvalue())
        self.assertIn("model_response", captured.getvalue())

    def test_handle_command_clear(self) -> None:
        """10. /clear command invokes agent.clear_logs()."""
        agent = ToolUsingAgent(
            model=MockModel(response="hello"),
            tool_registry=self.registry,
            event_collector=self.collector,
        )
        agent.send_message("test", session_id="clear-sess")
        self.assertEqual(len(agent.get_logs()), 3)

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cont = handle_command("/clear", agent, "clear-sess")
        self.assertTrue(cont)
        self.assertEqual(len(agent.get_logs()), 0)
        self.assertIn("Session logs cleared", captured.getvalue())

    def test_handle_command_reset(self) -> None:
        """11. /reset command invokes agent.reset()."""
        agent = ToolUsingAgent(
            model=MockModel(response="hello"),
            tool_registry=self.registry,
            event_collector=self.collector,
        )
        agent.send_message("test", session_id="reset-sess")

        captured = io.StringIO()
        with patch("sys.stdout", captured):
            cont = handle_command("/reset", agent, "reset-sess")
        self.assertTrue(cont)
        self.assertEqual(len(agent.get_logs()), 0)
        self.assertIn("reset", captured.getvalue().lower())

    def test_handle_command_quit_and_exit(self) -> None:
        """12. /quit and /exit return False to terminate session."""
        agent = ToolUsingAgent(model=MockModel(response=""), tool_registry=self.registry)
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            self.assertFalse(handle_command("/quit", agent, "sess"))
            self.assertFalse(handle_command("/exit", agent, "sess"))
        self.assertIn("Exiting KAS. Goodbye!", captured.getvalue())

    def test_setup_agent_initialization(self) -> None:
        """13. setup_agent initializes all controlled tools and agent with MockModel."""
        mock_model = MockModel(response="hello")
        agent, classifier, mock_server = setup_agent(
            model_adapter=mock_model,
            workspace_root=self.temp_dir,
        )
        try:
            self.assertIsInstance(agent, ToolUsingAgent)
            self.assertIsInstance(classifier, SecurityClassifier)
            tools = agent.tool_registry.list_tools()
            self.assertIn("calculator", tools)
            self.assertIn("file_tool", tools)
            self.assertIn("database_tool", tools)
            self.assertIn("http_tool", tools)
            self.assertIn("notification_tool", tools)
            self.assertIn("command_tool", tools)
        finally:
            if mock_server is not None:
                mock_server.stop()

    def test_setup_agent_default_timeout(self) -> None:
        """14. setup_agent defaults to 120.0s timeout."""
        import inspect
        sig = inspect.signature(setup_agent)
        self.assertEqual(sig.parameters["timeout"].default, 120.0)


if __name__ == "__main__":
    unittest.main()
