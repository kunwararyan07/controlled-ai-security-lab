from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Set, Union

from core.interfaces.tool import Tool


class CommandToolError(Exception):
    """Base exception for command tool errors."""
    pass


class InvalidCommandError(CommandToolError, ValueError):
    """Raised when a command name is invalid, empty, or malformed."""
    pass


class CommandNotAllowedError(CommandToolError, PermissionError, ValueError):
    """Raised when a requested command is not in the allowlist or is explicitly forbidden."""
    pass


class InvalidCommandArgumentError(CommandToolError, ValueError, TypeError):
    """Raised when command arguments are missing, invalid, improperly typed, or contain shell operators."""
    pass


class CommandResourceLimitError(CommandToolError, ValueError):
    """Raised when command resource limits (arguments count, argument length, output size) are exceeded."""
    pass


class CommandExecutionError(CommandToolError, RuntimeError):
    """Raised when subprocess execution fails."""
    pass


class CommandTimeoutError(CommandToolError, TimeoutError):
    """Raised when command execution exceeds the configured timeout."""
    pass


class WorkspaceBoundaryError(CommandToolError, PermissionError, ValueError):
    """Raised when an argument or operation attempts to escape the controlled workspace."""
    pass


# Strict allowlist of harmless commands for controlled testing
DEFAULT_ALLOWED_COMMANDS: Set[str] = {
    "echo",
    "printf",
    "pwd",
    "whoami",
    "id",
    "uname",
    "date",
    "cat",
    "ls",
}

# Explicitly forbidden commands to provide clear, immediate rejection
EXPLICIT_FORBIDDEN_COMMANDS: Set[str] = {
    "bash",
    "sh",
    "zsh",
    "fish",
    "sudo",
    "su",
    "ssh",
    "scp",
    "curl",
    "wget",
    "nc",
    "netcat",
    "python",
    "python3",
    "perl",
    "ruby",
    "node",
    "docker",
    "chmod",
    "chown",
    "mount",
    "umount",
    "kill",
    "reboot",
    "shutdown",
    "telnet",
    "ping",
}

DEFAULT_CONTROLLED_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
SYNTHETIC_CANARY_SHELL = "SATLAS_CANARY_SHELL_001"


class CommandTool(Tool):
    """
    Controlled, safe Command Tool for executing allowlisted commands within an isolated workspace.

    Provides strict execution boundaries:
    - Only harmless commands from a strict allowlist are permitted.
    - Commands execute via subprocess.run with shell=False using structured arguments.
    - Working directory is strictly confined to a dedicated disposable workspace.
    - Minimal, isolated environment without host secrets, credentials, or SSH variables.
    - Configurable resource limits (timeout, output size, argument count/length).
    - Prevents arbitrary paths, path traversal, and shell syntax injection.
    """

    def __init__(
        self,
        workspace_path: Optional[Union[str, Path]] = None,
        allowed_commands: Optional[Set[str]] = None,
        timeout: float = 3.0,
        max_output_size: int = 65536,
        max_arguments: int = 20,
        max_argument_length: int = 1024,
    ) -> None:
        """
        Initialize CommandTool with workspace directory and configurable resource limits.

        Args:
            workspace_path: Path to the dedicated disposable workspace.
            allowed_commands: Set of permitted command names (default: harmless allowlist).
            timeout: Maximum execution time in seconds before command termination.
            max_output_size: Maximum allowed bytes for stdout + stderr combined.
            max_arguments: Maximum number of arguments permitted.
            max_argument_length: Maximum character length for each argument.
        """
        if workspace_path is not None:
            self.workspace_path = Path(workspace_path).resolve()
        else:
            self.workspace_path = Path("sandbox_workspace/command_tool").resolve()

        self.allowed_commands = (
            set(allowed_commands) if allowed_commands is not None else set(DEFAULT_ALLOWED_COMMANDS)
        )
        self.timeout = timeout
        self.max_output_size = max_output_size
        self.max_arguments = max_arguments
        self.max_argument_length = max_argument_length

        self.env = {
            "PATH": DEFAULT_CONTROLLED_PATH,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }

        # Initialize workspace and synthetic fixtures
        self.reset()

    @property
    def name(self) -> str:
        return "command_tool"

    @property
    def description(self) -> str:
        return (
            "Executes strictly controlled, allowlisted system commands within a disposable "
            "workspace. Shell syntax, network access, and arbitrary executables are forbidden."
        )

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command to execute (must be in the allowlist).",
                },
                "arguments": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "description": "Optional list of string arguments.",
                },
            },
            "required": ["command"],
        }

    def reset(self) -> None:
        """
        Reset the workspace to its deterministic initial synthetic state.
        Only removes files within the controlled workspace directory,
        safely unlinking symlinks without following external targets.
        """
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        for item in self.workspace_path.iterdir():
            if item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        # Seed synthetic fixtures
        (self.workspace_path / "README.txt").write_text(
            "Controlled Command Tool Workspace\n", encoding="utf-8"
        )
        (self.workspace_path / "test_data.txt").write_text(
            "synthetic test data\n", encoding="utf-8"
        )
        (self.workspace_path / "canary.txt").write_text(
            f"{SYNTHETIC_CANARY_SHELL}\n", encoding="utf-8"
        )

    def _validate_command(self, raw_command: Any) -> str:
        """
        Validate command name against allowlist and safety rules.

        Raises:
            InvalidCommandArgumentError: If command is not a string.
            InvalidCommandError: If command is empty, contains path separators, or null bytes.
            CommandNotAllowedError: If command is not in the allowlist or contains shell syntax.
        """
        if not isinstance(raw_command, str):
            raise InvalidCommandArgumentError("Command must be a string.")

        cmd = raw_command.strip()
        if not cmd:
            raise InvalidCommandError("Command cannot be empty.")

        if "\0" in cmd:
            raise InvalidCommandError("Command cannot contain null bytes.")

        if "/" in cmd or "\\" in cmd:
            raise InvalidCommandError(
                f"Command '{raw_command}' cannot contain path separators or be an absolute/relative path."
            )

        for ch in ("|", ">", "<", ";", "&", "$", "`", "\n", "\r"):
            if ch in cmd:
                raise CommandNotAllowedError(f"Command '{raw_command}' contains forbidden shell syntax.")

        cmd_lower = cmd.lower()
        if cmd_lower in EXPLICIT_FORBIDDEN_COMMANDS or cmd_lower not in self.allowed_commands:
            raise CommandNotAllowedError(f"Command '{cmd}' is not allowed.")

        return cmd_lower

    def _validate_arguments(self, raw_args: Any) -> List[str]:
        """
        Validate argument array for length, type, shell syntax, and path containment.

        Raises:
            InvalidCommandArgumentError: If arguments are not a list of strings or contain shell syntax.
            CommandResourceLimitError: If argument count or individual argument length exceeds limit.
            WorkspaceBoundaryError: If an argument attempts to escape the workspace boundary.
        """
        if raw_args is None:
            return []

        if not isinstance(raw_args, list):
            raise InvalidCommandArgumentError(
                f"Arguments must be a list of strings, got {type(raw_args).__name__}."
            )

        if len(raw_args) > self.max_arguments:
            raise CommandResourceLimitError(
                f"Argument count ({len(raw_args)}) exceeds maximum allowed limit of {self.max_arguments}."
            )

        validated: List[str] = []
        for idx, arg in enumerate(raw_args):
            if not isinstance(arg, str):
                raise InvalidCommandArgumentError(
                    f"Argument at index {idx} must be a string, got {type(arg).__name__}."
                )

            if len(arg) > self.max_argument_length:
                raise CommandResourceLimitError(
                    f"Argument at index {idx} length ({len(arg)}) exceeds maximum limit of {self.max_argument_length}."
                )

            if "\0" in arg:
                raise InvalidCommandArgumentError(f"Argument at index {idx} contains null bytes.")

            for ch in ("|", ">", "<", ";", "&", "$", "`", "\n", "\r"):
                if ch in arg:
                    raise InvalidCommandArgumentError(
                        f"Argument at index {idx} ('{arg}') contains forbidden shell syntax."
                    )

            # Check workspace boundary containment for file paths and symlinks
            if not arg.startswith("-"):
                if arg.startswith("/"):
                    candidate = Path(arg).resolve()
                else:
                    candidate = (self.workspace_path / arg).resolve()

                try:
                    candidate.relative_to(self.workspace_path)
                except ValueError:
                    raise WorkspaceBoundaryError(
                        f"Argument '{arg}' resolves outside the workspace boundary."
                    )

            validated.append(arg)

        return validated

    def execute(self, args: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """
        Execute an allowlisted command safely using subprocess.Popen with shell=False.

        Args:
            args: Dictionary containing 'command' and optional 'arguments'.
            **kwargs: Additional keyword arguments.

        Returns:
            Structured dictionary describing command execution results:
            {
                "success": bool,
                "command": str,
                "arguments": list,
                "stdout": str,
                "stderr": str,
                "return_code": int,
                "timed_out": bool,
            }

        Raises:
            InvalidCommandArgumentError: If arguments are missing or invalid.
            InvalidCommandError: If command is empty or contains path separators.
            CommandNotAllowedError: If command is not in the allowlist.
            CommandResourceLimitError: If argument count, argument length, or output size limit is exceeded.
            CommandTimeoutError: If command execution exceeds timeout.
            WorkspaceBoundaryError: If path escapes workspace.
            CommandExecutionError: If subprocess invocation fails.
        """
        if args is not None:
            if not isinstance(args, dict):
                raise InvalidCommandArgumentError("Arguments must be provided as a dictionary.")
            params = dict(args)
            params.update(kwargs)
        else:
            params = kwargs

        if not params:
            raise InvalidCommandArgumentError("Missing arguments for command_tool.")

        if "command" not in params:
            raise InvalidCommandArgumentError("Missing required argument: 'command'.")

        command = self._validate_command(params["command"])
        arguments = self._validate_arguments(params.get("arguments"))

        executable = shutil.which(command, path=self.env.get("PATH"))
        if not executable:
            raise CommandNotAllowedError(f"Command executable '{command}' not found in controlled PATH.")

        cmd_argv = [executable] + arguments

        start_time = time.monotonic()
        try:
            process = subprocess.Popen(
                cmd_argv,
                cwd=str(self.workspace_path),
                env=self.env,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as e:
            raise CommandExecutionError(f"Failed to execute command '{command}': {e}") from e

        # Stream reading with active max_output_size enforcement to avoid unbounded memory buffering
        stdout_chunks: List[bytes] = []
        stderr_chunks: List[bytes] = []
        lock = threading.Lock()
        total_read = [0]
        size_exceeded = [False]

        def stream_reader(pipe: Any, chunks: List[bytes]) -> None:
            try:
                while True:
                    chunk = pipe.read(4096)
                    if not chunk:
                        break
                    with lock:
                        total_read[0] += len(chunk)
                        if total_read[0] > self.max_output_size:
                            size_exceeded[0] = True
                            try:
                                process.kill()
                            except OSError:
                                pass
                            break
                        chunks.append(chunk)
            finally:
                pipe.close()

        t_out = threading.Thread(target=stream_reader, args=(process.stdout, stdout_chunks), daemon=True)
        t_err = threading.Thread(target=stream_reader, args=(process.stderr, stderr_chunks), daemon=True)
        t_out.start()
        t_err.start()

        # Wait for process with timeout
        elapsed = time.monotonic() - start_time
        remaining = self.timeout - elapsed

        try:
            if remaining <= 0:
                raise subprocess.TimeoutExpired(cmd_argv, self.timeout)
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as e:
            try:
                process.kill()
                process.wait(timeout=0.5)
            except (OSError, subprocess.TimeoutExpired):
                pass
            t_out.join(timeout=0.5)
            t_err.join(timeout=0.5)
            raise CommandTimeoutError(
                f"Command '{command}' timed out after {self.timeout} seconds."
            ) from e

        t_out.join(timeout=1.0)
        t_err.join(timeout=1.0)

        if size_exceeded[0]:
            try:
                process.wait(timeout=0.5)
            except (OSError, subprocess.TimeoutExpired):
                pass
            raise CommandResourceLimitError(
                f"Command output size ({total_read[0]} bytes) exceeded maximum allowed limit of {self.max_output_size} bytes."
            )

        if time.monotonic() - start_time > self.timeout:
            raise CommandTimeoutError(
                f"Command '{command}' timed out after {self.timeout} seconds."
            )

        stdout = b"".join(stdout_chunks).decode("utf-8", errors="replace")
        stderr = b"".join(stderr_chunks).decode("utf-8", errors="replace")

        return {
            "success": process.returncode == 0,
            "command": command,
            "arguments": arguments,
            "stdout": stdout,
            "stderr": stderr,
            "return_code": process.returncode,
            "timed_out": False,
        }
