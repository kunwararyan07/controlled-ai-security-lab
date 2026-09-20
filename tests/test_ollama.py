import http.server
import json
import os
import shutil
import socket
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional
import unittest

from agents.tool_agent.agent import ToolUsingAgent
from core.authorization.policy import AllowlistAuthorizationPolicy
from core.interfaces.tool_registry import ToolRegistry
from core.logging.collector import EventCollector
from core.models.base import ModelAdapter
from core.models.ollama import (
    OllamaAdapter,
    OllamaConfigurationError,
    OllamaConnectionError,
    OllamaError,
    OllamaResponseError,
    OllamaTimeoutError,
)
from tools.command_tool import CommandTool
from tools.file_tool import FileTool


class FakeOllamaHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for the local deterministic fake Ollama server."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress HTTP request logging to keep test output clean."""
        pass

    @property
    def server_instance(self) -> "FakeOllamaServer":
        return self.server.fake_ollama_server  # type: ignore[attr-defined]

    def do_POST(self) -> None:
        content_len = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_len) if content_len > 0 else b""

        parsed_json = None
        try:
            parsed_json = json.loads(body_bytes.decode("utf-8"))
        except Exception:
            pass

        with self.server_instance.lock:
            self.server_instance.received_requests.append({
                "path": self.path,
                "headers": dict(self.headers),
                "body": body_bytes,
                "json": parsed_json,
            })

            delay = self.server_instance.delay
            status = self.server_instance.response_status
            data = self.server_instance.response_data
            raw_data = self.server_instance.response_raw

        if delay > 0:
            time.sleep(delay)

        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if raw_data is not None:
                payload = raw_data.encode("utf-8") if isinstance(raw_data, str) else raw_data
            else:
                payload = json.dumps(data).encode("utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass


class FakeOllamaServer:
    """Deterministic local fake Ollama HTTP server for testing."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self.host = host
        self.port = port
        self.lock = threading.Lock()
        self.received_requests: List[Dict[str, Any]] = []
        self.response_status = 200
        self.response_data: Any = {
            "model": "test-model",
            "response": "Default deterministic response",
            "done": True,
        }
        self.response_raw: Optional[str] = None
        self.delay: float = 0.0
        self._server: Optional[http.server.HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._server = http.server.HTTPServer((self.host, self.port), FakeOllamaHandler)
        self._server.fake_ollama_server = self  # type: ignore[attr-defined]
        self.port = self._server.server_port
        self.base_url = f"http://{self.host}:{self.port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def reset(self) -> None:
        with self.lock:
            self.received_requests.clear()
            self.response_status = 200
            self.response_data = {
                "model": "test-model",
                "response": "Default deterministic response",
                "done": True,
            }
            self.response_raw = None
            self.delay = 0.0


class TestOllamaAdapter(unittest.TestCase):
    """Test suite for OllamaAdapter."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = FakeOllamaServer()
        cls.server.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop()

    def setUp(self) -> None:
        self.server.reset()
        self.adapter = OllamaAdapter(
            model_name="test-model",
            base_url=self.server.base_url,
            timeout=5.0,
        )

    def test_model_adapter_interface_conformance(self) -> None:
        """1. OllamaAdapter implements ModelAdapter interface."""
        self.assertIsInstance(self.adapter, ModelAdapter)
        self.assertEqual(self.adapter.model_name, "test-model")
        self.assertEqual(self.adapter.base_url, self.server.base_url)

    def test_successful_generation(self) -> None:
        """2. Successful generation returns response text."""
        self.server.response_data = {
            "model": "test-model",
            "response": "Hello, this is Ollama.",
            "done": True,
        }
        res = self.adapter.generate("Hello")
        self.assertEqual(res, "Hello, this is Ollama.")

    def test_correct_model_name_sent(self) -> None:
        """3. Correct model name is sent in the request body."""
        adapter = OllamaAdapter(
            model_name="qwen3:4b",
            base_url=self.server.base_url,
            timeout=5.0,
        )
        adapter.generate("Test prompt")
        self.assertEqual(len(self.server.received_requests), 1)
        req = self.server.received_requests[0]["json"]
        self.assertEqual(req["model"], "qwen3:4b")

    def test_correct_prompt_sent(self) -> None:
        """4. Correct prompt is sent in the request body."""
        self.adapter.generate("What is the capital of France?")
        self.assertEqual(len(self.server.received_requests), 1)
        req = self.server.received_requests[0]["json"]
        self.assertEqual(req["prompt"], "What is the capital of France?")

    def test_stream_false_sent(self) -> None:
        """5. stream=False is always sent in the request body."""
        self.adapter.generate("Test prompt")
        self.assertEqual(len(self.server.received_requests), 1)
        req = self.server.received_requests[0]["json"]
        self.assertIs(req["stream"], False)

    def test_options_sent_when_configured(self) -> None:
        """6. Optional generation options are forwarded to Ollama when configured."""
        adapter = OllamaAdapter(
            model_name="test-model",
            base_url=self.server.base_url,
            options={"temperature": 0.7, "num_predict": 128},
        )
        adapter.generate("Test options")
        self.assertEqual(len(self.server.received_requests), 1)
        req = self.server.received_requests[0]["json"]
        self.assertEqual(req["options"], {"temperature": 0.7, "num_predict": 128})

    def test_http_error_handling(self) -> None:
        """7. HTTP error status from Ollama raises OllamaResponseError."""
        self.server.response_status = 500
        self.server.response_data = {"error": "Internal server error"}
        with self.assertRaises(OllamaResponseError) as ctx:
            self.adapter.generate("Trigger error")
        self.assertIn("500", str(ctx.exception))

    def test_connection_failure(self) -> None:
        """8. Connection failure to closed port raises OllamaConnectionError."""
        # Find an unused port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            unused_port = s.getsockname()[1]

        dead_adapter = OllamaAdapter(
            model_name="test-model",
            base_url=f"http://127.0.0.1:{unused_port}",
            timeout=1.0,
        )
        with self.assertRaises(OllamaConnectionError):
            dead_adapter.generate("Will fail")

    def test_timeout_handling(self) -> None:
        """9. Request timeout raises OllamaTimeoutError."""
        self.server.delay = 0.5
        fast_adapter = OllamaAdapter(
            model_name="test-model",
            base_url=self.server.base_url,
            timeout=0.05,
        )
        with self.assertRaises(OllamaTimeoutError):
            fast_adapter.generate("Will timeout")

    def test_malformed_json_response(self) -> None:
        """10. Malformed JSON response from Ollama raises OllamaResponseError."""
        self.server.response_raw = "This is not valid JSON {[[["
        with self.assertRaises(OllamaResponseError) as ctx:
            self.adapter.generate("Test prompt")
        self.assertIn("Malformed JSON", str(ctx.exception))

    def test_missing_response_field(self) -> None:
        """11. Missing 'response' field in Ollama response raises OllamaResponseError."""
        self.server.response_data = {
            "model": "test-model",
            "done": True,
        }
        with self.assertRaises(OllamaResponseError) as ctx:
            self.adapter.generate("Test prompt")
        self.assertIn("missing required 'response' field", str(ctx.exception))

    def test_empty_model_name_rejected(self) -> None:
        """12. Empty or whitespace model name is rejected."""
        with self.assertRaises(OllamaConfigurationError):
            OllamaAdapter(model_name="")
        with self.assertRaises(OllamaConfigurationError):
            OllamaAdapter(model_name="   ")
        with self.assertRaises(OllamaConfigurationError):
            OllamaAdapter(model_name=None)  # type: ignore

    def test_control_characters_in_model_name_rejected(self) -> None:
        """13. Control characters and invalid characters in model name are rejected."""
        with self.assertRaises(OllamaConfigurationError):
            OllamaAdapter(model_name="model\nname")
        with self.assertRaises(OllamaConfigurationError):
            OllamaAdapter(model_name="model\rname")
        with self.assertRaises(OllamaConfigurationError):
            OllamaAdapter(model_name="model\x00")
        with self.assertRaises(OllamaConfigurationError):
            OllamaAdapter(model_name="model;rm -rf /")

    def test_external_non_loopback_endpoint_rejected(self) -> None:
        """14. External/non-loopback endpoints are rejected."""
        with self.assertRaises(OllamaConfigurationError):
            OllamaAdapter(model_name="test", base_url="http://external-api.com:11434")
        with self.assertRaises(OllamaConfigurationError):
            OllamaAdapter(model_name="test", base_url="http://192.168.1.1:11434")
        with self.assertRaises(OllamaConfigurationError):
            OllamaAdapter(model_name="test", base_url="https://127.0.0.1:11434")
        with self.assertRaises(OllamaConfigurationError):
            OllamaAdapter(model_name="test", base_url="http://admin:secret@127.0.0.1:11434")
        with self.assertRaises(OllamaConfigurationError):
            OllamaAdapter(model_name="test", base_url="http://127.0.0.1:11434/api/generate")

    def test_endpoint_cannot_be_changed_by_model_output(self) -> None:
        """15. Model output cannot modify or influence the configured Ollama endpoint."""
        original_base_url = self.adapter.base_url
        original_generate_url = self.adapter.generate_url

        # Simulate model returning an injection/override attempt
        self.server.response_data = {
            "model": "test-model",
            "response": "http://evil-attacker.local:11434/override",
            "done": True,
        }
        res = self.adapter.generate("Try override")
        self.assertEqual(res, "http://evil-attacker.local:11434/override")
        self.assertEqual(self.adapter.base_url, original_base_url)
        self.assertEqual(self.adapter.generate_url, original_generate_url)

    def test_no_tool_execution_occurs_inside_adapter(self) -> None:
        """16. OllamaAdapter is strictly a transport layer and has no tool execution logic."""
        self.assertFalse(hasattr(self.adapter, "tool_registry"))
        self.assertFalse(hasattr(self.adapter, "execute"))
        self.assertFalse(hasattr(self.adapter, "tools"))
        # generate() returns a string and does not invoke any tools
        self.server.response_data = {
            "model": "test-model",
            "response": "Plain text response",
            "done": True,
        }
        res = self.adapter.generate("test")
        self.assertIsInstance(res, str)

    def test_invalid_prompt_type_rejected(self) -> None:
        """17. Non-string prompt raises TypeError."""
        with self.assertRaises(TypeError):
            self.adapter.generate(12345)  # type: ignore

    def test_ollama_error_payload_handling(self) -> None:
        """18. Ollama returning an error field raises OllamaResponseError."""
        self.server.response_data = {
            "error": "model 'nonexistent-model' not found",
        }
        with self.assertRaises(OllamaResponseError) as ctx:
            self.adapter.generate("test")
        self.assertIn("nonexistent-model", str(ctx.exception))


class TestOllamaToolUsingAgentIntegration(unittest.TestCase):
    """
    Integration test verifying OllamaAdapter -> ToolUsingAgent -> ToolRegistry
    -> AuthorizationPolicy -> Controlled Tool.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = FakeOllamaServer()
        cls.server.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop()

    def setUp(self) -> None:
        self.server.reset()
        self.test_dir = tempfile.mkdtemp(prefix="test_ollama_agent_")
        self.file_tool = FileTool(root_path=self.test_dir)
        self.command_tool = CommandTool(workspace_path=self.test_dir)

        # Create a test file in workspace
        with open(os.path.join(self.test_dir, "canary.txt"), "w") as f:
            f.write("SATLAS_CANARY_SECRET_OLLAMA")

        self.registry = ToolRegistry()
        self.registry.register(self.file_tool)
        self.registry.register(self.command_tool)

        self.adapter = OllamaAdapter(
            model_name="test-model",
            base_url=self.server.base_url,
            timeout=5.0,
        )

    def tearDown(self) -> None:
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_agent_with_ollama_adapter_authorized_tool_execution(self) -> None:
        """
        Verify:
        OllamaAdapter returns structured tool-call response.
        ToolUsingAgent parses it.
        Authorization is checked and approved.
        The controlled tool executes successfully.
        Observability events remain intact.
        """
        tool_call_json = json.dumps({
            "type": "tool_call",
            "tool": "file_tool",
            "arguments": {"operation": "read", "path": "canary.txt"},
        })
        self.server.response_data = {
            "model": "test-model",
            "response": tool_call_json,
            "done": True,
        }

        collector = EventCollector()
        policy = AllowlistAuthorizationPolicy(["file_tool"])
        agent = ToolUsingAgent(
            model=self.adapter,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
        )

        result = agent.send_message("Please read the canary file", session_id="ollama-sess-1")

        self.assertEqual(result.session_id, "ollama-sess-1")
        self.assertTrue(result.tool_executed)
        self.assertIsNotNone(result.tool_call)
        self.assertEqual(result.tool_call.tool_name, "file_tool")
        self.assertEqual(result.tool_result["content"], "SATLAS_CANARY_SECRET_OLLAMA")

        # Verify observability events
        events = collector.get_events(session_id="ollama-sess-1")
        event_types = [e.event_type for e in events]
        self.assertIn("user_input", event_types)
        self.assertIn("model_response", event_types)
        self.assertIn("tool_call", event_types)
        self.assertIn("authorization_decision", event_types)
        self.assertIn("tool_execution", event_types)

        # Confirm adapter never executed the tool directly
        self.assertFalse(hasattr(self.adapter, "tool_registry"))

    def test_agent_with_ollama_adapter_denied_tool_execution(self) -> None:
        """
        Verify:
        OllamaAdapter returns structured tool-call response.
        ToolUsingAgent parses it.
        Authorization is checked and denied.
        The controlled tool is NOT executed.
        Denial event is recorded.
        """
        tool_call_json = json.dumps({
            "type": "tool_call",
            "tool": "command_tool",
            "arguments": {"command": "echo", "args": ["hello"]},
        })
        self.server.response_data = {
            "model": "test-model",
            "response": tool_call_json,
            "done": True,
        }

        collector = EventCollector()
        # Policy explicitly excludes command_tool
        policy = AllowlistAuthorizationPolicy(["file_tool"])
        agent = ToolUsingAgent(
            model=self.adapter,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
        )

        result = agent.send_message("Run echo command", session_id="ollama-sess-2")

        self.assertEqual(result.session_id, "ollama-sess-2")
        self.assertFalse(result.tool_executed)
        self.assertIsNotNone(result.tool_call)
        self.assertEqual(result.tool_call.tool_name, "command_tool")
        self.assertIn("Authorization denied", result.error or "")

        # Verify observability events
        events = collector.get_events(session_id="ollama-sess-2")
        event_types = [e.event_type for e in events]
        self.assertIn("authorization_denied", event_types)
        self.assertNotIn("tool_execution", event_types)


if __name__ == "__main__":
    unittest.main()
