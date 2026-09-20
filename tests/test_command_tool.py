from pathlib import Path
import tempfile
import unittest

from core.interfaces.tool import Tool
from tools.command_tool import (
    CommandExecutionError,
    CommandNotAllowedError,
    CommandResourceLimitError,
    CommandTimeoutError,
    CommandTool,
    CommandToolError,
    InvalidCommandArgumentError,
    InvalidCommandError,
    SYNTHETIC_CANARY_SHELL,
    WorkspaceBoundaryError,
)


class TestCommandTool(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name).resolve()
        self.tool = CommandTool(workspace_path=self.workspace)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_tool_interface_conformance(self):
        """1. Tool interface conformance."""
        self.assertIsInstance(self.tool, Tool)
        self.assertEqual(self.tool.name, "command_tool")
        self.assertIsInstance(self.tool.description, str)
        self.assertTrue(len(self.tool.description) > 0)

    def test_tool_metadata_and_schema(self):
        """2. Metadata and schema."""
        metadata = self.tool.get_metadata()
        self.assertEqual(metadata["name"], "command_tool")
        self.assertEqual(metadata["description"], self.tool.description)
        self.assertIn("properties", metadata["schema"])
        self.assertIn("command", metadata["schema"]["properties"])
        self.assertIn("arguments", metadata["schema"]["properties"])

    def test_allowed_echo_command(self):
        """3. Allowed echo command."""
        result = self.tool.execute({
            "command": "echo",
            "arguments": ["hello", "world"],
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["command"], "echo")
        self.assertEqual(result["arguments"], ["hello", "world"])
        self.assertEqual(result["stdout"], "hello world\n")
        self.assertEqual(result["stderr"], "")
        self.assertEqual(result["return_code"], 0)
        self.assertFalse(result["timed_out"])

    def test_allowed_pwd_command(self):
        """4. Allowed pwd command reports workspace path."""
        result = self.tool.execute({"command": "pwd"})
        self.assertTrue(result["success"])
        self.assertEqual(result["stdout"].strip(), str(self.workspace))

    def test_allowed_ls_command(self):
        """5. Allowed ls command lists workspace contents."""
        result = self.tool.execute({"command": "ls"})
        self.assertTrue(result["success"])
        output = result["stdout"]
        self.assertIn("README.txt", output)
        self.assertIn("test_data.txt", output)
        self.assertIn("canary.txt", output)

    def test_allowed_cat_of_synthetic_file(self):
        """6. Allowed cat of synthetic test file."""
        result = self.tool.execute({
            "command": "cat",
            "arguments": ["test_data.txt"],
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["stdout"], "synthetic test data\n")

    def test_synthetic_canary_file(self):
        """7. Synthetic canary file retrieval via cat."""
        result = self.tool.execute({
            "command": "cat",
            "arguments": ["canary.txt"],
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["stdout"].strip(), SYNTHETIC_CANARY_SHELL)

    def test_deterministic_workspace_reset(self):
        """8. Deterministic workspace reset."""
        # Create an extra file
        extra = self.workspace / "extra.txt"
        extra.write_text("extra data\n", encoding="utf-8")
        self.assertTrue(extra.exists())

        # Reset workspace
        self.tool.reset()

        # Extra file must be gone, synthetic fixtures restored
        self.assertFalse(extra.exists())
        self.assertTrue((self.workspace / "README.txt").exists())
        self.assertTrue((self.workspace / "test_data.txt").exists())
        self.assertTrue((self.workspace / "canary.txt").exists())
        self.assertEqual(
            (self.workspace / "canary.txt").read_text(encoding="utf-8").strip(),
            SYNTHETIC_CANARY_SHELL,
        )

    def test_deterministic_command_behavior(self):
        """9. Deterministic command behavior across multiple runs."""
        r1 = self.tool.execute({"command": "echo", "arguments": ["deterministic"]})
        r2 = self.tool.execute({"command": "echo", "arguments": ["deterministic"]})
        self.assertEqual(r1["stdout"], r2["stdout"])
        self.assertEqual(r1["return_code"], r2["return_code"])

    def test_reject_unknown_command(self):
        """10. Reject unknown command."""
        with self.assertRaises(CommandNotAllowedError):
            self.tool.execute({"command": "unknown_cmd"})

    def test_reject_bash(self):
        """11. Reject bash."""
        with self.assertRaises(CommandNotAllowedError):
            self.tool.execute({"command": "bash"})

    def test_reject_sh(self):
        """12. Reject sh."""
        with self.assertRaises(CommandNotAllowedError):
            self.tool.execute({"command": "sh"})

    def test_reject_sudo(self):
        """13. Reject sudo."""
        with self.assertRaises(CommandNotAllowedError):
            self.tool.execute({"command": "sudo", "arguments": ["ls"]})

    def test_reject_ssh(self):
        """14. Reject ssh."""
        with self.assertRaises(CommandNotAllowedError):
            self.tool.execute({"command": "ssh", "arguments": ["user@host"]})

    def test_reject_curl(self):
        """15. Reject curl."""
        with self.assertRaises(CommandNotAllowedError):
            self.tool.execute({"command": "curl", "arguments": ["http://example.com"]})

    def test_reject_wget(self):
        """16. Reject wget."""
        with self.assertRaises(CommandNotAllowedError):
            self.tool.execute({"command": "wget", "arguments": ["http://example.com"]})

    def test_reject_arbitrary_executable_path(self):
        """17. Reject relative or arbitrary executable paths."""
        with self.assertRaises(InvalidCommandError):
            self.tool.execute({"command": "../../bin/cat"})

    def test_reject_absolute_executable_path(self):
        """18. Reject absolute executable paths."""
        with self.assertRaises(InvalidCommandError):
            self.tool.execute({"command": "/bin/echo"})

    def test_reject_malformed_command_arguments(self):
        """19. Reject non-list or non-string arguments."""
        with self.assertRaises(InvalidCommandArgumentError):
            self.tool.execute({"command": "echo", "arguments": "not-a-list"})

        with self.assertRaises(InvalidCommandArgumentError):
            self.tool.execute({"command": "echo", "arguments": [123]})

        with self.assertRaises(InvalidCommandArgumentError):
            self.tool.execute({"command": "echo", "arguments": [{"bad": "dict"}]})

    def test_reject_shell_syntax_in_arguments(self):
        """20. Reject shell operators / injection in arguments."""
        shell_injections = [
            "hello | cat",
            "file > output",
            "$(whoami)",
            "`whoami`",
            "hello && whoami",
            "hello; whoami",
            "hello || whoami",
            "hello\nwhoami",
        ]
        for injection in shell_injections:
            with self.assertRaises(InvalidCommandArgumentError):
                self.tool.execute({"command": "echo", "arguments": [injection]})

    def test_reject_null_bytes(self):
        """21. Reject null bytes in command and arguments."""
        with self.assertRaises(InvalidCommandError):
            self.tool.execute({"command": "echo\x00"})

        with self.assertRaises(InvalidCommandArgumentError):
            self.tool.execute({"command": "echo", "arguments": ["hello\x00"]})

    def test_argument_count_limit(self):
        """22. Reject argument count exceeding limit."""
        limited_tool = CommandTool(workspace_path=self.workspace, max_arguments=3)
        with self.assertRaises(CommandResourceLimitError):
            limited_tool.execute({
                "command": "echo",
                "arguments": ["1", "2", "3", "4"],
            })

    def test_argument_length_limit(self):
        """23. Reject argument length exceeding limit."""
        limited_tool = CommandTool(workspace_path=self.workspace, max_argument_length=10)
        with self.assertRaises(CommandResourceLimitError):
            limited_tool.execute({
                "command": "echo",
                "arguments": ["A" * 11],
            })

    def test_output_size_limit(self):
        """24. Reject commands exceeding maximum output size."""
        limited_tool = CommandTool(workspace_path=self.workspace, max_output_size=10)
        with self.assertRaises(CommandResourceLimitError):
            limited_tool.execute({
                "command": "echo",
                "arguments": ["This output exceeds 10 bytes easily"],
            })

    def test_timeout_handling(self):
        """25. Timeout handling terminates long-running commands."""
        # Extremely small timeout ensures timeout triggers
        short_timeout_tool = CommandTool(workspace_path=self.workspace, timeout=0.000001)
        with self.assertRaises(CommandTimeoutError):
            short_timeout_tool.execute({
                "command": "ls",
            })

    def test_workspace_traversal_protection(self):
        """26. Workspace traversal protection blocks escaping arguments."""
        traversal_attempts = [
            "/etc/passwd",
            "/",
            "../../",
            "../../etc/shadow",
            "subdir/../../../../etc/passwd",
        ]
        for target in traversal_attempts:
            with self.assertRaises(WorkspaceBoundaryError):
                self.tool.execute({"command": "cat", "arguments": [target]})

    def test_cannot_choose_arbitrary_cwd(self):
        """27. Working directory is strictly fixed to the workspace."""
        # Attempting to supply a cwd parameter is ignored or rejected
        res = self.tool.execute({"command": "pwd", "cwd": "/tmp"})
        self.assertEqual(res["stdout"].strip(), str(self.workspace))

    def test_minimal_environment_behavior(self):
        """28. Subprocess does not expose host environment secrets."""
        # Ensure tool env only contains controlled variables
        self.assertEqual(set(self.tool.env.keys()), {"PATH", "LANG", "LC_ALL"})
        self.assertNotIn("HOME", self.tool.env)
        self.assertNotIn("SSH_AUTH_SOCK", self.tool.env)

    def test_no_network_oriented_commands(self):
        """29. Network-oriented commands are strictly forbidden."""
        network_commands = [
            "nc",
            "netcat",
            "telnet",
            "ping",
            "curl",
            "wget",
            "ssh",
            "scp",
        ]
        for cmd in network_commands:
            with self.assertRaises(CommandNotAllowedError):
                self.tool.execute({"command": cmd})

    def test_reset_behavior(self):
        """30. Reset clears non-fixture files."""
        # Create temp file
        temp_file = self.workspace / "temp.log"
        temp_file.write_text("temporary log", encoding="utf-8")
        self.assertTrue(temp_file.exists())

        self.tool.reset()
        self.assertFalse(temp_file.exists())
        self.assertTrue((self.workspace / "canary.txt").exists())

    def test_multiple_sequential_commands(self):
        """31. Multiple sequential commands execute safely."""
        r1 = self.tool.execute({"command": "echo", "arguments": ["first"]})
        self.assertEqual(r1["stdout"], "first\n")

        r2 = self.tool.execute({"command": "cat", "arguments": ["test_data.txt"]})
        self.assertEqual(r2["stdout"], "synthetic test data\n")

        r3 = self.tool.execute({"command": "ls"})
        self.assertIn("README.txt", r3["stdout"])

    def test_keyword_arguments_execution(self):
        """Execution using direct keyword arguments."""
        result = self.tool.execute(
            command="echo",
            arguments=["keyword", "args"],
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["stdout"], "keyword args\n")

    def test_output_size_exactly_at_limit(self):
        """32. Output exactly at max_output_size limit succeeds."""
        limited_tool = CommandTool(workspace_path=self.workspace, max_output_size=10)
        result = limited_tool.execute({
            "command": "echo",
            "arguments": ["-n", "0123456789"],
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["stdout"], "0123456789")
        self.assertEqual(len(result["stdout"].encode("utf-8")), 10)

    def test_output_size_exceeding_limit_actively_terminated(self):
        """33. Output exceeding limit is actively terminated and raises CommandResourceLimitError."""
        limited_tool = CommandTool(workspace_path=self.workspace, max_output_size=10)
        with self.assertRaises(CommandResourceLimitError):
            limited_tool.execute({
                "command": "echo",
                "arguments": ["0123456789 extra bytes"],
            })

    def test_symlink_pointing_outside_workspace_rejected(self):
        """34. Symlink pointing outside workspace is rejected by workspace containment check."""
        with tempfile.TemporaryDirectory() as outside_dir:
            outside_file = Path(outside_dir) / "secret.txt"
            outside_file.write_text("external secret", encoding="utf-8")

            symlink_path = self.workspace / "symlink_to_outside"
            symlink_path.symlink_to(outside_file)

            with self.assertRaises(WorkspaceBoundaryError):
                self.tool.execute({"command": "cat", "arguments": ["symlink_to_outside"]})

    def test_internal_symlink_and_legitimate_file_allowed(self):
        """35. Legitimate files and internal symlinks within workspace execute successfully."""
        # Internal symlink to canary fixture
        symlink_internal = self.workspace / "symlink_to_canary.txt"
        symlink_internal.symlink_to(self.workspace / "canary.txt")

        result = self.tool.execute({"command": "cat", "arguments": ["symlink_to_canary.txt"]})
        self.assertTrue(result["success"])
        self.assertEqual(result["stdout"].strip(), SYNTHETIC_CANARY_SHELL)

        # Direct legitimate file
        result2 = self.tool.execute({"command": "cat", "arguments": ["test_data.txt"]})
        self.assertTrue(result2["success"])
        self.assertEqual(result2["stdout"], "synthetic test data\n")

    def test_reset_removes_symlinks_without_deleting_external_target(self):
        """36. Reset removes symlinks without following or deleting external targets."""
        with tempfile.TemporaryDirectory() as outside_dir:
            outside_file = Path(outside_dir) / "target.txt"
            outside_file.write_text("external data", encoding="utf-8")

            outside_subdir = Path(outside_dir) / "target_dir"
            outside_subdir.mkdir()
            (outside_subdir / "file.txt").write_text("dir content", encoding="utf-8")

            # Create symlinks inside workspace pointing to external file and dir
            symlink_file = self.workspace / "link_file"
            symlink_file.symlink_to(outside_file)

            symlink_dir = self.workspace / "link_dir"
            symlink_dir.symlink_to(outside_subdir)

            self.assertTrue(symlink_file.is_symlink())
            self.assertTrue(symlink_dir.is_symlink())

            self.tool.reset()

            # Symlinks in workspace must be removed
            self.assertFalse(symlink_file.exists())
            self.assertFalse(symlink_dir.exists())
            self.assertFalse(symlink_file.is_symlink())
            self.assertFalse(symlink_dir.is_symlink())

            # External targets must still exist untouched
            self.assertTrue(outside_file.exists())
            self.assertEqual(outside_file.read_text(encoding="utf-8"), "external data")
            self.assertTrue(outside_subdir.exists())
            self.assertTrue((outside_subdir / "file.txt").exists())


if __name__ == "__main__":
    unittest.main()

