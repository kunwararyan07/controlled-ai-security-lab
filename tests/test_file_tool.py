from pathlib import Path
import tempfile
import unittest

from core.interfaces.tool import Tool
from tools.file_tool import (
    FileSizeLimitExceededError,
    FileTool,
    FileToolError,
    FileNotFoundError,
    InvalidArgumentError,
    InvalidOperationError,
    SandboxViolationError,
)


class TestFileTool(unittest.TestCase):
    def setUp(self):
        # Create an isolated temporary directory as sandbox root for each test
        self.temp_dir = tempfile.TemporaryDirectory()
        self.sandbox_root = Path(self.temp_dir.name).resolve()
        self.file_tool = FileTool(root_path=self.sandbox_root)

        # Also create a separate temporary directory outside the sandbox for escape tests
        self.outside_dir = tempfile.TemporaryDirectory()
        self.outside_root = Path(self.outside_dir.name).resolve()

    def tearDown(self):
        self.temp_dir.cleanup()
        self.outside_dir.cleanup()

    def test_tool_metadata_and_schema(self):
        """1. Tool metadata and schema exist and conform to Tool interface."""
        self.assertIsInstance(self.file_tool, Tool)
        self.assertEqual(self.file_tool.name, "file_tool")
        self.assertIsInstance(self.file_tool.description, str)
        self.assertTrue(len(self.file_tool.description) > 0)
        self.assertIn("properties", self.file_tool.schema)
        self.assertIn("operation", self.file_tool.schema["properties"])
        self.assertIn("path", self.file_tool.schema["properties"])
        metadata = self.file_tool.get_metadata()
        self.assertEqual(metadata["name"], "file_tool")
        self.assertEqual(metadata["schema"], self.file_tool.schema)

    def test_list_sandbox_directory(self):
        """2. List files and directories inside the sandbox."""
        # Create some test files and directories inside sandbox
        (self.sandbox_root / "file1.txt").write_text("hello 1", encoding="utf-8")
        (self.sandbox_root / "file2.txt").write_text("hello 2", encoding="utf-8")
        (self.sandbox_root / "subdir").mkdir()
        (self.sandbox_root / "subdir" / "nested.txt").write_text("nested", encoding="utf-8")

        result = self.file_tool.execute({"operation": "list", "path": "."})

        self.assertEqual(result["operation"], "list")
        self.assertEqual(result["count"], 3)
        entry_names = [e["name"] for e in result["entries"]]
        self.assertIn("file1.txt", entry_names)
        self.assertIn("file2.txt", entry_names)
        self.assertIn("subdir", entry_names)

        # Test listing subdirectory
        sub_result = self.file_tool.execute({"operation": "list", "path": "subdir"})
        self.assertEqual(sub_result["count"], 1)
        self.assertEqual(sub_result["entries"][0]["name"], "nested.txt")
        self.assertEqual(sub_result["entries"][0]["type"], "file")

    def test_list_symlink_pointing_outside_sandbox(self):
        """Regression test: symlink pointing outside sandbox is listed safely without following or inspecting target."""
        # Create an external directory with sensitive files outside sandbox
        external_dir = self.outside_root / "external_secret_dir"
        external_dir.mkdir()
        (external_dir / "external_file.txt").write_text("outside data", encoding="utf-8")

        # Create an external file outside sandbox
        external_file = self.outside_root / "external_secret_file.txt"
        external_file.write_text("outside file data", encoding="utf-8")

        # Create symlinks inside the sandbox pointing to external targets
        symlink_to_dir = self.sandbox_root / "symlink_dir"
        symlink_to_file = self.sandbox_root / "symlink_file.txt"
        try:
            symlink_to_dir.symlink_to(external_dir)
            symlink_to_file.symlink_to(external_file)
        except OSError:
            self.skipTest("Symlink creation not supported on this platform")

        # 1. Listing the sandbox root should safely report the symlinks without following them
        result = self.file_tool.execute({"operation": "list", "path": "."})
        entries_by_name = {e["name"]: e for e in result["entries"]}

        self.assertIn("symlink_dir", entries_by_name)
        self.assertTrue(entries_by_name["symlink_dir"]["is_symlink"])
        self.assertEqual(entries_by_name["symlink_dir"]["type"], "symlink")

        self.assertIn("symlink_file.txt", entries_by_name)
        self.assertTrue(entries_by_name["symlink_file.txt"]["is_symlink"])
        self.assertEqual(entries_by_name["symlink_file.txt"]["type"], "symlink")

        # Verify no external content (such as 'external_file.txt') is enumerated
        entry_names = list(entries_by_name.keys())
        self.assertNotIn("external_file.txt", entry_names)

        # 2. Attempting to traverse/list into the symlinked external directory must be rejected
        with self.assertRaises(SandboxViolationError):
            self.file_tool.execute({"operation": "list", "path": "symlink_dir"})

    def test_read_allowed_file(self):
        """3. Read an allowed UTF-8 text file inside the sandbox."""
        test_file = self.sandbox_root / "public.txt"
        test_file.write_text("Hello from sandbox!", encoding="utf-8")

        result = self.file_tool.execute({"operation": "read", "path": "public.txt"})

        self.assertEqual(result["operation"], "read")
        self.assertEqual(result["path"], "public.txt")
        self.assertEqual(result["content"], "Hello from sandbox!")
        self.assertEqual(result["size"], len("Hello from sandbox!"))

    def test_write_allowed_file(self):
        """4. Write an allowed UTF-8 text file inside the sandbox."""
        result = self.file_tool.execute({
            "operation": "write",
            "path": "test.txt",
            "content": "hello world",
        })

        self.assertEqual(result["operation"], "write")
        self.assertEqual(result["path"], "test.txt")
        self.assertTrue(result["success"])
        self.assertEqual(result["bytes_written"], len(b"hello world"))

        # Verify on filesystem inside sandbox
        created_file = self.sandbox_root / "test.txt"
        self.assertTrue(created_file.exists())
        self.assertEqual(created_file.read_text(encoding="utf-8"), "hello world")

        # Test writing with nested parent directory creation
        nested_result = self.file_tool.execute({
            "operation": "write",
            "path": "deep/nested/dir/note.txt",
            "content": "deep content",
        })
        self.assertTrue(nested_result["success"])
        self.assertTrue((self.sandbox_root / "deep/nested/dir/note.txt").exists())

    def test_reject_absolute_path_outside_sandbox(self):
        """5. Reject absolute path outside sandbox."""
        # Target an outside file
        outside_file = self.outside_root / "secret.txt"
        outside_file.write_text("outside data", encoding="utf-8")

        with self.assertRaises(SandboxViolationError):
            self.file_tool.execute({"operation": "read", "path": str(outside_file)})

        with self.assertRaises(SandboxViolationError):
            self.file_tool.execute({"operation": "write", "path": str(outside_file), "content": "bad"})

        # Try host system paths
        with self.assertRaises(SandboxViolationError):
            self.file_tool.execute({"operation": "read", "path": "/etc/passwd"})

    def test_reject_dot_dot_traversal(self):
        """6. Reject ../ path traversal."""
        with self.assertRaises(SandboxViolationError):
            self.file_tool.execute({"operation": "read", "path": "../outside.txt"})

        with self.assertRaises(SandboxViolationError):
            self.file_tool.execute({"operation": "read", "path": "../../etc/passwd"})

        # Subdirectory traversal escaping root
        (self.sandbox_root / "sub").mkdir()
        with self.assertRaises(SandboxViolationError):
            self.file_tool.execute({"operation": "read", "path": "sub/../../outside.txt"})

    def test_reject_normalized_or_encoded_traversal(self):
        """7. Reject traversal using normalized or complex path representations."""
        with self.assertRaises(SandboxViolationError):
            self.file_tool.execute({"operation": "read", "path": "foo/./../../outside.txt"})

        with self.assertRaises(SandboxViolationError):
            self.file_tool.execute({"operation": "read", "path": "a/b/../../../outside.txt"})

    def test_reject_symlink_escape_outside_sandbox(self):
        """8. Reject symlink inside sandbox pointing outside sandbox."""
        outside_file = self.outside_root / "host_sensitive.txt"
        outside_file.write_text("host secret", encoding="utf-8")

        # Create a symlink inside the sandbox pointing to the outside file
        symlink_path = self.sandbox_root / "symlink_to_outside.txt"
        try:
            symlink_path.symlink_to(outside_file)
        except OSError:
            self.skipTest("Symlink creation not supported on this platform")

        # Attempting to read via the symlink must be blocked
        with self.assertRaises(SandboxViolationError):
            self.file_tool.execute({"operation": "read", "path": "symlink_to_outside.txt"})

        # Attempting to write via the symlink must be blocked
        with self.assertRaises(SandboxViolationError):
            self.file_tool.execute({
                "operation": "write",
                "path": "symlink_to_outside.txt",
                "content": "overwrite attempt",
            })

        # Outside file must remain untouched
        self.assertEqual(outside_file.read_text(encoding="utf-8"), "host secret")

    def test_reject_reading_non_existent_file(self):
        """9. Reject reading a non-existent file cleanly."""
        with self.assertRaises(FileNotFoundError):
            self.file_tool.execute({"operation": "read", "path": "missing_file.txt"})

    def test_reject_writing_outside_sandbox(self):
        """10. Reject writing outside sandbox."""
        with self.assertRaises(SandboxViolationError):
            self.file_tool.execute({
                "operation": "write",
                "path": "../escaped.txt",
                "content": "malicious content",
            })

        # Ensure no file was created outside sandbox
        escaped_file = self.sandbox_root.parent / "escaped.txt"
        self.assertFalse(escaped_file.exists())

    def test_enforce_maximum_read_and_write_size(self):
        """11. Enforce maximum read/write size limits."""
        small_tool = FileTool(root_path=self.sandbox_root, max_read_size=50, max_write_size=50)

        # Test write size limit
        large_content = "A" * 60
        with self.assertRaises(FileSizeLimitExceededError):
            small_tool.execute({
                "operation": "write",
                "path": "large.txt",
                "content": large_content,
            })

        # Write an allowed file directly on filesystem, then test read size limit
        large_file = self.sandbox_root / "large_read.txt"
        large_file.write_text("B" * 60, encoding="utf-8")

        with self.assertRaises(FileSizeLimitExceededError):
            small_tool.execute({"operation": "read", "path": "large_read.txt"})

    def test_synthetic_canary_file_behavior(self):
        """12. Verify synthetic canary files inside sandbox can be accessed when authorized."""
        fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "tool_tests" / "canary_secrets.env"
        self.assertTrue(fixture_path.exists(), "Synthetic canary fixture must exist in fixtures/tool_tests/")
        canary_content = fixture_path.read_text(encoding="utf-8")

        # Copy fixture content to a file inside the sandbox
        canary_file = self.sandbox_root / "canary_secrets.env"
        canary_file.write_text(canary_content, encoding="utf-8")

        # Read canary file
        read_result = self.file_tool.execute({"operation": "read", "path": "canary_secrets.env"})
        self.assertIn("SATLAS_CANARY_SECRET_001", read_result["content"])
        self.assertIn("SATLAS_CANARY_USER_001", read_result["content"])
        self.assertIn("SATLAS_INTERNAL_TOKEN_001", read_result["content"])

        # Write canary update inside sandbox
        update_result = self.file_tool.execute({
            "operation": "write",
            "path": "canary_output.txt",
            "content": "VALIDATED_SATLAS_CANARY_SECRET_001",
        })
        self.assertTrue(update_result["success"])
        self.assertEqual(
            (self.sandbox_root / "canary_output.txt").read_text(encoding="utf-8"),
            "VALIDATED_SATLAS_CANARY_SECRET_001",
        )

    def test_no_host_filesystem_access_outside_sandbox(self):
        """13. Verify no host filesystem access occurs outside the configured sandbox."""
        # Attempt listing parent directory
        with self.assertRaises(SandboxViolationError):
            self.file_tool.execute({"operation": "list", "path": ".."})

        # Attempt listing root directory
        with self.assertRaises(SandboxViolationError):
            self.file_tool.execute({"operation": "list", "path": "/"})

    def test_invalid_arguments_and_operations(self):
        """Test validation of missing/invalid arguments and operations."""
        with self.assertRaises(InvalidArgumentError):
            self.file_tool.execute({})

        with self.assertRaises(InvalidArgumentError):
            self.file_tool.execute({"path": "file.txt"})

        with self.assertRaises(InvalidArgumentError):
            self.file_tool.execute({"operation": "read"})

        with self.assertRaises(InvalidArgumentError):
            self.file_tool.execute({"operation": "read", "path": ""})

        with self.assertRaises(InvalidOperationError):
            self.file_tool.execute({"operation": "delete", "path": "file.txt"})


if __name__ == "__main__":
    unittest.main()
