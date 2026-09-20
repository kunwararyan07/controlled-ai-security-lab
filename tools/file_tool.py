from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from core.interfaces.tool import Tool


class FileToolError(Exception):
    """Base exception for file tool errors."""
    pass


class SandboxViolationError(FileToolError, PermissionError, ValueError):
    """Raised when an operation attempts to access paths outside the sandbox."""
    pass


class FileNotFoundError(FileToolError, OSError):
    """Raised when a requested file or directory is not found."""
    pass


class FileSizeLimitExceededError(FileToolError, ValueError):
    """Raised when a file exceeds the configured size limit."""
    pass


class InvalidOperationError(FileToolError, ValueError):
    """Raised when an unsupported operation is requested."""
    pass


class InvalidArgumentError(FileToolError, ValueError, TypeError):
    """Raised when arguments are invalid or missing."""
    pass


class FileTool(Tool):
    """
    Controlled, sandboxed file tool for reading, writing, and listing files.

    All operations are strictly confined within a configured sandbox root directory.
    Attempts to escape via absolute paths, '../' traversals, or symlinks are rejected.
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        max_read_size: int = 1024 * 1024,   # 1 MB default
        max_write_size: int = 1024 * 1024,  # 1 MB default
    ) -> None:
        """
        Initialize the FileTool with a sandbox root directory.

        Args:
            root_path: Directory path acting as the boundary for all file operations.
            max_read_size: Maximum allowed bytes to read in a single operation.
            max_write_size: Maximum allowed bytes to write in a single operation.
        """
        self.root_path = Path(root_path).resolve()
        self.max_read_size = max_read_size
        self.max_write_size = max_write_size

        # Ensure sandbox root exists
        if not self.root_path.exists():
            self.root_path.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "file_tool"

    @property
    def description(self) -> str:
        return (
            "Performs sandboxed file operations (list, read, write) strictly "
            "confined within a designated root directory."
        )

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["list", "read", "write"],
                    "description": "The file operation to perform (list, read, write).",
                },
                "path": {
                    "type": "string",
                    "description": "Path to the target file or directory relative to the sandbox root.",
                },
                "content": {
                    "type": "string",
                    "description": "UTF-8 text content to write (required for write operation).",
                },
            },
            "required": ["operation", "path"],
        }

    def _resolve_safe_path(self, path: Union[str, Path], must_exist: bool = False) -> Path:
        """
        Canonicalize the given path and verify it stays inside the sandbox root.

        Args:
            path: Input path (relative or absolute).
            must_exist: Whether the target path must already exist.

        Returns:
            Resolved, canonical Path guaranteed to be inside self.root_path.

        Raises:
            InvalidArgumentError: If path is empty.
            SandboxViolationError: If resolved path escapes sandbox root.
            FileNotFoundError: If must_exist is True and path does not exist.
        """
        if not path or not str(path).strip():
            raise InvalidArgumentError("Path cannot be empty.")

        p = Path(path)

        # If path is absolute, resolve directly; if relative, resolve from root_path
        if p.is_absolute():
            candidate = p.resolve()
        else:
            candidate = (self.root_path / p).resolve()

        # Strict containment check against canonical sandbox root
        try:
            candidate.relative_to(self.root_path)
        except ValueError:
            raise SandboxViolationError(
                f"Access denied: Path '{path}' resolves to '{candidate}', "
                f"which is outside the sandbox root '{self.root_path}'."
            )

        if must_exist and not candidate.exists():
            raise FileNotFoundError(f"File or directory not found: '{path}'")

        return candidate

    def execute(self, args: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """
        Execute the requested file operation.

        Args:
            args: Dictionary containing 'operation', 'path', and optional 'content'.
            **kwargs: Can also be passed as keyword arguments.

        Returns:
            Structured dictionary describing operation result.

        Raises:
            InvalidArgumentError: If arguments are missing or invalid.
            InvalidOperationError: If operation is not list, read, or write.
            SandboxViolationError: If path escapes sandbox.
            FileNotFoundError: If target path does not exist when required.
            FileSizeLimitExceededError: If file or content exceeds size limits.
        """
        if args is not None:
            if not isinstance(args, dict):
                raise InvalidArgumentError("Arguments must be provided as a dictionary.")
            params = dict(args)
            params.update(kwargs)
        else:
            params = kwargs

        if not params:
            raise InvalidArgumentError("Missing arguments for file_tool.")

        if "operation" not in params:
            raise InvalidArgumentError("Missing required argument: 'operation'.")

        operation = params["operation"]
        if not isinstance(operation, str):
            raise InvalidArgumentError("Argument 'operation' must be a string.")

        op = operation.strip().lower()

        if "path" not in params:
            raise InvalidArgumentError("Missing required argument: 'path'.")

        path = params["path"]
        if not isinstance(path, (str, Path)):
            raise InvalidArgumentError("Argument 'path' must be a string or Path.")

        if op == "list":
            return self._handle_list(path)
        elif op == "read":
            return self._handle_read(path)
        elif op == "write":
            content = params.get("content")
            return self._handle_write(path, content)
        else:
            raise InvalidOperationError(
                f"Unsupported operation: '{operation}'. Supported operations: list, read, write."
            )

    def _handle_list(self, path: Union[str, Path]) -> Dict[str, Any]:
        """List files and directories inside the sandbox path."""
        safe_path = self._resolve_safe_path(path, must_exist=True)

        if not safe_path.is_dir():
            raise FileToolError(f"Path is not a directory: '{path}'")

        entries: List[Dict[str, Any]] = []
        for item in sorted(safe_path.iterdir(), key=lambda p: p.name):
            is_symlink = item.is_symlink()
            if is_symlink:
                # Do not follow symlink to inspect external targets or expose external metadata
                entry_type = "symlink"
                size = item.lstat().st_size
            else:
                is_dir = item.is_dir()
                entry_type = "directory" if is_dir else "file"
                size = item.stat().st_size if not is_dir else 0

            rel_path = str(item.relative_to(self.root_path))
            entries.append({
                "name": item.name,
                "path": rel_path,
                "type": entry_type,
                "size": size,
                "is_symlink": is_symlink,
            })

        rel_dir = str(safe_path.relative_to(self.root_path))
        return {
            "operation": "list",
            "path": rel_dir if rel_dir != "." else "",
            "entries": entries,
            "count": len(entries),
        }

    def _handle_read(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Read a UTF-8 text file inside the sandbox."""
        safe_path = self._resolve_safe_path(path, must_exist=True)

        if safe_path.is_dir():
            raise FileToolError(f"Cannot read directory as file: '{path}'")

        size = safe_path.stat().st_size
        if size > self.max_read_size:
            raise FileSizeLimitExceededError(
                f"File size {size} bytes exceeds maximum allowed read size of {self.max_read_size} bytes."
            )

        try:
            content = safe_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise FileToolError(f"Cannot read file '{path}': UTF-8 decoding failed: {str(e)}")

        return {
            "operation": "read",
            "path": str(safe_path.relative_to(self.root_path)),
            "content": content,
            "size": len(content),
        }

    def _handle_write(self, path: Union[str, Path], content: Optional[Any]) -> Dict[str, Any]:
        """Write UTF-8 text to a file inside the sandbox."""
        if content is None or not isinstance(content, str):
            raise InvalidArgumentError("Write operation requires 'content' string argument.")

        content_bytes = content.encode("utf-8")
        if len(content_bytes) > self.max_write_size:
            raise FileSizeLimitExceededError(
                f"Content size {len(content_bytes)} bytes exceeds maximum allowed write size of {self.max_write_size} bytes."
            )

        safe_path = self._resolve_safe_path(path, must_exist=False)

        if safe_path.is_dir():
            raise FileToolError(f"Cannot write to existing directory: '{path}'")

        # Create parent directories safely inside the sandbox
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")

        return {
            "operation": "write",
            "path": str(safe_path.relative_to(self.root_path)),
            "success": True,
            "bytes_written": len(content_bytes),
        }
