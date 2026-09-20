# Controlled AI Security Lab

A controlled security testing laboratory for evaluating AI agent tool execution boundaries, authorization enforcement, and model-facing tool-use protocols in an isolated environment.

---

## KAS Tool-Calling Agent — Interactive Manual Runner

The repository includes **KAS**, a dedicated interactive manual CLI runner for testing the `ToolUsingAgent` with local Ollama models and controlled tools.

```text
╔════════════════════════════════════════════════════════════╗
║                 KAS TOOL_CALLING AGENT                    ║
╚════════════════════════════════════════════════════════════╝
```

### Starting the KAS CLI

To start the interactive session with the default model (`gemma2:2b`):

```bash
python3 scripts/run_tool_agent.py
```

To specify an alternative local Ollama model (e.g. `tinyllama:latest` or `qwen3:4b`):

```bash
python3 scripts/run_tool_agent.py --model tinyllama:latest
```

To configure a custom local Ollama endpoint or adjust timeout:

```bash
python3 scripts/run_tool_agent.py --model gemma2:2b --endpoint http://127.0.0.1:11434 --timeout 60.0 --max-steps 3
```

*(Note: Endpoint must point to localhost/loopback; external network endpoints are strictly rejected).*

---

### Interactive Commands

During a KAS session, the following slash commands are available:

| Command | Description |
|---|---|
| `/help` | Display the list of available commands and usage instructions. |
| `/state` | Display current agent lifecycle state, registered tools, and configuration. |
| `/logs` | Display structured observability events recorded for the current session. |
| `/clear` | Clear recorded session logs from the event collector. |
| `/reset` | Reset agent lifecycle state, event collector logs, and all tool fixtures. |
| `/quit` (or `/exit`) | Terminate the KAS interactive session cleanly. |

---

### Example Interaction

```text
You > Calculate 25 * 4 using the calculator tool.

────────────────────────────────────────────────────────────
  KAS RESPONSE & TOOL EXECUTION REPORT
────────────────────────────────────────────────────────────
KAS > 100

  Tool Requested:         YES
  Tool Name:              calculator
  Tool Arguments:         {"operation": "multiply", "a": 25, "b": 4}
  Authorization Decision: ALLOWED
  Execution Status:       EXECUTED
  Tool Result:            100
  Security Status:        PASS - Normal safe interaction; no security boundary was crossed.
────────────────────────────────────────────────────────────
```
