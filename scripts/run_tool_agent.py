#!/usr/bin/env python3
"""
KAS Tool-Calling Agent — Interactive Manual Runner

Provides an interactive CLI interface for testing and interacting with the
ToolUsingAgent using a real local Ollama model.
"""

import argparse
import json
import os
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.tool_agent.agent import ToolUsingAgent, create_full_tool_registry
from agents.tool_agent.result import AgentResult
from core.authorization.policy import AllowlistAuthorizationPolicy
from core.interfaces.tool_registry import ToolRegistry
from core.logging.collector import EventCollector
from core.models.base import ModelAdapter
from core.models.ollama import OllamaAdapter
from core.security.classifier import SecurityClassifier
from tools.calculator import CalculatorTool
from tools.command_tool import CommandTool
from tools.database_tool import DatabaseTool
from tools.file_tool import FileTool
from tools.http_tool import HTTPTool
from tools.mock_api_server import MockAPIServer
from tools.notification_tool import NotificationTool

BANNER = """\
╔════════════════════════════════════════════════════════════╗
║                 KAS TOOL_CALLING AGENT                    ║
╚════════════════════════════════════════════════════════════╝\
"""

DEFAULT_ALLOWED_TOOLS = [
    "calculator",
    "file_tool",
    "database_tool",
    "http_tool",
    "notification_tool",
    "command_tool",
]


def format_banner() -> str:
    """Return the prominent KAS branding banner."""
    return BANNER


def format_agent_result(result: AgentResult, classifier: SecurityClassifier) -> str:
    """
    Format an AgentResult into a clear, structured console report.

    Displays:
    - Final response
    - Tool requested status
    - Tool name and arguments
    - Authorization decision
    - Execution status
    - Tool result
    - Errors, if any
    - Security classification
    """
    lines: List[str] = []
    lines.append("")
    lines.append("─" * 60)
    lines.append("  KAS RESPONSE & TOOL EXECUTION REPORT")
    lines.append("─" * 60)

    # 1. Final response
    if result.final_response is not None and result.final_response.strip():
        lines.append(f"KAS > {result.final_response.strip()}")
    else:
        lines.append("KAS > [No final text response]")

    # 2. Tool call details
    if result.tool_call is not None:
        lines.append("")
        lines.append("  Tool Requested:         YES")
        lines.append(f"  Tool Name:              {result.tool_call.tool_name}")
        lines.append(f"  Tool Arguments:         {json.dumps(result.tool_call.arguments)}")

        if result.authorization_allowed:
            auth_str = "ALLOWED"
        elif result.authorization_denied:
            auth_str = f"DENIED ({result.authorization_reason or 'Policy rejection'})"
        else:
            auth_str = "N/A"
        lines.append(f"  Authorization Decision: {auth_str}")

        exec_str = "EXECUTED" if result.tool_executed else "NOT EXECUTED"
        lines.append(f"  Execution Status:       {exec_str}")

        if result.tool_executed and result.tool_result is not None:
            lines.append(f"  Tool Result:            {json.dumps(result.tool_result, default=str)}")
    else:
        lines.append("")
        lines.append("  Tool Requested:         NO")

    # 3. Multi-step summary if multiple tools were invoked
    if len(result.tool_calls) > 1:
        lines.append(f"  Total Tool Calls:       {len(result.tool_calls)}")
        lines.append("  Execution History:")
        for idx, tc in enumerate(result.tool_calls, 1):
            if idx - 1 < len(result.tool_results):
                status_desc = f"EXECUTED -> {json.dumps(result.tool_results[idx - 1], default=str)}"
            else:
                status_desc = "NOT EXECUTED"
            lines.append(f"    [{idx}] {tc.tool_name}: {status_desc}")

    # 4. Errors
    if result.error:
        lines.append(f"  Error:                  {result.error}")

    # 5. Security classification
    sec_result = classifier.classify_agent_result(result)
    lines.append(f"  Security Status:        {sec_result.status.value} - {sec_result.reason}")

    # 6. Timing metrics (if available)
    if result.total_duration is not None or result.model_durations:
        lines.append("")
        lines.append("  Timing Details:")
        if result.total_duration is not None:
            lines.append(f"    Total Interaction:    {result.total_duration:.3f}s")
        for idx, md in enumerate(result.model_durations, 1):
            lines.append(f"    Model Generation #{idx}: {md:.3f}s")
        for idx, td in enumerate(result.tool_durations, 1):
            tool_name = result.tool_calls[idx - 1].tool_name if idx - 1 < len(result.tool_calls) else f"Tool #{idx}"
            lines.append(f"    {tool_name} Execution: {td:.3f}s")

    lines.append("─" * 60)
    lines.append("")
    return "\n".join(lines)


def handle_command(
    command_line: str,
    agent: ToolUsingAgent,
    session_id: str,
) -> bool:
    """
    Handle slash commands (/help, /state, /logs, /clear, /reset, /quit).

    Returns:
        True to continue the REPL session, False to exit.
    """
    cmd = command_line.strip().lower()

    if cmd in ("/quit", "/exit"):
        print("\n[Exiting KAS. Goodbye!]")
        return False

    elif cmd == "/help":
        print("""
Available KAS Commands:
  /help    Show this help message
  /state   Display current agent lifecycle and configuration state
  /logs    Display structured observability events for this session
  /clear   Clear recorded events from the event collector
  /reset   Reset agent lifecycle state, logs, and tool fixtures
  /quit    Exit the KAS interactive session (or /exit)
""")
        return True

    elif cmd == "/state":
        state = agent.get_state()
        print("\n--- AGENT STATE ---")
        print(json.dumps(state, indent=2))
        print("-------------------\n")
        return True

    elif cmd == "/logs":
        events = agent.get_logs(session_id=session_id)
        if not events:
            print("\n[No events recorded for this session yet.]\n")
        else:
            print(f"\n--- SESSION LOGS ({session_id}) ---")
            for i, e in enumerate(events, 1):
                payload_info = ""
                if e.tool_call:
                    payload_info = f" | Tool: {e.tool_call}"
                elif e.error:
                    payload_info = f" | Error: {e.error}"
                if e.duration_seconds is not None:
                    payload_info += f" | Duration: {e.duration_seconds:.3f}s"
                print(f"[{i}] {e.timestamp} | Type: {e.event_type} | State: {e.agent_state}{payload_info}")
            print("----------------------------------\n")
        return True

    elif cmd == "/clear":
        agent.clear_logs()
        print("\n[Session logs cleared from event collector.]\n")
        return True

    elif cmd == "/reset":
        agent.reset()
        print("\n[Agent lifecycle state, event logs, and tool fixtures have been reset.]\n")
        return True

    else:
        print(f"\n[Unknown command '{command_line}'. Type /help for available commands.]\n")
        return True


DEFAULT_KAS_MODEL_OPTIONS: Dict[str, Any] = {
    "temperature": 0.0,
    "num_predict": 256,
}


def setup_agent(
    model_name: str = "gemma2:2b",
    base_url: str = "http://127.0.0.1:11434",
    timeout: float = 120.0,
    model_adapter: Optional[ModelAdapter] = None,
    allowed_tools: Optional[List[str]] = None,
    max_steps: int = 3,
    workspace_root: Optional[str] = None,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[ToolUsingAgent, SecurityClassifier, Optional[MockAPIServer]]:
    """
    Initialize and return ToolUsingAgent, SecurityClassifier, and optional MockAPIServer.

    Uses existing OllamaAdapter and ToolUsingAgent with strictly confined tools.
    """
    root = workspace_root or os.path.join(REPO_ROOT, "sandbox_workspace")
    file_ws = os.path.join(root, "file_sandbox")
    db_path = os.path.join(root, "database_tool", "test.db")
    cmd_ws = os.path.join(root, "command_tool")

    os.makedirs(file_ws, exist_ok=True)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    os.makedirs(cmd_ws, exist_ok=True)

    # Initialize controlled tools
    file_tool = FileTool(root_path=file_ws)
    database_tool = DatabaseTool(database_path=db_path)

    # Local synthetic mock API server for HTTPTool
    mock_server = MockAPIServer()
    mock_server.start()
    http_tool = HTTPTool(base_url=mock_server.base_url)

    notification_tool = NotificationTool()
    command_tool = CommandTool(workspace_path=cmd_ws)
    calculator_tool = CalculatorTool()

    # Tool registry
    registry = ToolRegistry()
    registry.register(calculator_tool)
    registry.register(file_tool)
    registry.register(database_tool)
    registry.register(http_tool)
    registry.register(notification_tool)
    registry.register(command_tool)

    # Authorization policy
    tools_to_allow = allowed_tools or list(DEFAULT_ALLOWED_TOOLS)
    policy = AllowlistAuthorizationPolicy(allowed_tools=tools_to_allow)

    # Observability and classification
    collector = EventCollector()
    classifier = SecurityClassifier()

    # Model adapter
    model_options = options if options is not None else dict(DEFAULT_KAS_MODEL_OPTIONS)
    model = model_adapter or OllamaAdapter(
        model_name=model_name,
        base_url=base_url,
        timeout=timeout,
        options=model_options,
    )

    agent = ToolUsingAgent(
        model=model,
        tool_registry=registry,
        authorization_policy=policy,
        event_collector=collector,
        max_steps=max_steps,
    )

    return agent, classifier, mock_server


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="KAS Tool-Calling Agent — Interactive Manual Runner",
    )
    parser.add_argument(
        "--model",
        default="gemma2:2b",
        help="Local Ollama model name to use (default: gemma2:2b)",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:11434",
        help="Ollama base URL (default: http://127.0.0.1:11434, must point to localhost)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Model request timeout in seconds (default: 120.0)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3,
        help="Maximum tool execution steps per interaction (default: 3)",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session ID to reuse (default: auto-generated)",
    )
    return parser.parse_args(args)


def run_repl(
    agent: ToolUsingAgent,
    classifier: SecurityClassifier,
    session_id: Optional[str] = None,
    mock_server: Optional[MockAPIServer] = None,
) -> None:
    """Run the interactive REPL loop."""
    sid = session_id or f"kas-{uuid.uuid4().hex[:8]}"
    print(format_banner())
    model_name = getattr(agent.model, "model_name", type(agent.model).__name__)
    print(f"\n[Session: {sid} | Model: {model_name}]")
    print("Type /help for available commands or /quit to exit.\n")

    try:
        while True:
            try:
                user_input = input("You > ").strip()
            except EOFError:
                print("\n[Exiting KAS. Goodbye!]")
                break
            except KeyboardInterrupt:
                print("\n[Session interrupted. Type /quit or press Ctrl+D to exit.]")
                continue

            if not user_input:
                continue

            if user_input.startswith("/"):
                should_continue = handle_command(user_input, agent, sid)
                if not should_continue:
                    break
                continue

            try:
                result = agent.send_message(user_input, session_id=sid)
                print(format_agent_result(result, classifier))
            except Exception as e:
                print(f"\n[Execution error: {e}]\n")

    finally:
        if mock_server is not None:
            mock_server.stop()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    try:
        agent, classifier, mock_server = setup_agent(
            model_name=args.model,
            base_url=args.endpoint,
            timeout=args.timeout,
            max_steps=args.max_steps,
        )
    except Exception as e:
        print(f"Failed to initialize KAS agent: {e}", file=sys.stderr)
        sys.exit(1)

    run_repl(agent, classifier, session_id=args.session_id, mock_server=mock_server)


if __name__ == "__main__":
    main()
