import json
import posixpath
from typing import Any, Dict, Optional, Union
import urllib.error
import urllib.parse
import urllib.request

from core.interfaces.tool import Tool


class HTTPToolError(Exception):
    """Base exception for HTTP tool errors."""
    pass


class NetworkSafetyViolationError(HTTPToolError, PermissionError, ValueError):
    """Raised when an operation violates network safety constraints."""
    pass


class InvalidMethodError(HTTPToolError, ValueError):
    """Raised when an unsupported HTTP method is requested."""
    pass


class InvalidPathError(HTTPToolError, ValueError):
    """Raised when an invalid relative API path is provided."""
    pass


class InvalidArgumentError(HTTPToolError, ValueError, TypeError):
    """Raised when arguments are missing, invalid, or improperly typed."""
    pass


class HTTPExecutionError(HTTPToolError, RuntimeError):
    """Raised when HTTP execution fails (e.g., connection refused, timeout)."""
    pass


class ResourceLimitExceededError(HTTPToolError, ValueError):
    """Base exception raised when an operation exceeds configured resource limits."""
    pass


class PayloadSizeLimitExceededError(ResourceLimitExceededError):
    """Raised when request body exceeds maximum allowed size."""
    pass


class ResponseSizeLimitExceededError(ResourceLimitExceededError):
    """Raised when response body exceeds maximum allowed size."""
    pass


class StrictRedirectHandler(urllib.request.HTTPRedirectHandler):
    """
    HTTP redirect handler that enforces strict network boundaries.
    Only permits redirects that remain strictly within the configured local base URL.
    """

    def __init__(self, allowed_base_url: str) -> None:
        self.allowed_base = urllib.parse.urlsplit(allowed_base_url)

    def redirect_request(
        self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Optional[urllib.request.Request]:
        target = urllib.parse.urlsplit(newurl)
        if target.scheme != self.allowed_base.scheme or target.netloc != self.allowed_base.netloc:
            raise NetworkSafetyViolationError(
                f"External redirect blocked: Target '{newurl}' escapes allowed base URL."
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class HTTPTool(Tool):
    """
    Controlled, safe HTTP Tool for interacting exclusively with a local synthetic mock API.

    Provides strict network safety boundaries:
    - Communicates ONLY with a configured local mock server (127.0.0.1 or localhost).
    - Arbitrary full URLs, external domains, non-HTTP schemes, and external redirects are strictly blocked.
    - Supports ONLY GET and POST methods.
    - Enforces JSON serialization for POST bodies without code evaluation.
    - Enforces maximum request body and response size limits.
    """

    SUPPORTED_METHODS = {"GET", "POST"}

    def __init__(
        self,
        base_url: str,
        max_body_size: int = 65536,         # 64 KB default
        max_response_size: int = 1048576,   # 1 MB default
        timeout: float = 5.0,               # 5 seconds default
    ) -> None:
        """
        Initialize the HTTPTool with a validated local mock server base URL.

        Args:
            base_url: Base URL of the local synthetic mock server (e.g. 'http://127.0.0.1:8080').
            max_body_size: Maximum allowed bytes for request body.
            max_response_size: Maximum allowed bytes for response body.
            timeout: Timeout in seconds for HTTP requests.
        """
        if not isinstance(base_url, str) or not base_url.strip():
            raise NetworkSafetyViolationError("base_url must be a non-empty string.")

        parsed = urllib.parse.urlsplit(base_url.strip())
        if parsed.scheme != "http":
            raise NetworkSafetyViolationError(
                f"Only HTTP scheme is supported for mock API, got '{parsed.scheme}'."
            )

        hostname = parsed.hostname
        if hostname not in ("127.0.0.1", "localhost"):
            raise NetworkSafetyViolationError(
                f"Mock API base_url must point to localhost (127.0.0.1 or localhost), got '{hostname}'."
            )

        self.base_url = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        self.max_body_size = max_body_size
        self.max_response_size = max_response_size
        self.timeout = timeout

        # Build custom opener with strict redirect handling
        self._opener = urllib.request.build_opener(StrictRedirectHandler(self.base_url))

    @property
    def name(self) -> str:
        return "http_tool"

    @property
    def description(self) -> str:
        return (
            "Performs controlled, safe HTTP operations (GET, POST) against a "
            "configured local synthetic mock API."
        )

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST"],
                    "description": "HTTP method to execute (GET or POST only).",
                },
                "path": {
                    "type": "string",
                    "description": "Relative API path (e.g. '/health', '/users', '/records').",
                },
                "body": {
                    "type": ["object", "array", "string", "number", "boolean", "null"],
                    "description": "Optional request body (for POST). Must be JSON-serializable.",
                },
                "headers": {
                    "type": "object",
                    "description": "Optional HTTP request headers.",
                },
            },
            "required": ["method", "path"],
        }

    def _validate_path(self, path: Any) -> str:
        """
        Validate that the path is a clean relative API path and does not attempt escape.

        Args:
            path: Relative API path string.

        Returns:
            Validated, normalized relative API path.

        Raises:
            InvalidArgumentError: If path is missing or not a string.
            InvalidPathError: If path is malformed.
            NetworkSafetyViolationError: If path attempts to specify an external host or scheme.
        """
        if not isinstance(path, str):
            raise InvalidArgumentError("Argument 'path' must be a string.")

        cleaned = path.strip()
        if not cleaned:
            raise InvalidArgumentError("Path cannot be empty.")

        if "\\" in cleaned:
            raise InvalidPathError("Path cannot contain backslashes.")

        # Check for scheme, host, or authority components
        parsed = urllib.parse.urlsplit(cleaned)
        if parsed.scheme or parsed.netloc:
            raise NetworkSafetyViolationError(
                f"External destination or full URL rejected: '{path}'. Only relative paths are permitted."
            )

        if cleaned.startswith("//"):
            raise NetworkSafetyViolationError(
                f"Scheme-relative URLs rejected: '{path}'."
            )

        # Normalize path
        norm_path = posixpath.normpath(parsed.path)
        if not norm_path.startswith("/"):
            norm_path = "/" + norm_path

        # Preserve query string if present
        if parsed.query:
            return f"{norm_path}?{parsed.query}"
        return norm_path

    def execute(self, args: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """
        Execute the requested HTTP operation against the local mock server.

        Args:
            args: Dictionary containing 'method', 'path', and optional 'body', 'headers'.
            **kwargs: Additional keyword arguments forwarded or passed directly.

        Returns:
            Structured dictionary with 'status_code', 'headers', 'body', and 'success'.

        Raises:
            InvalidArgumentError: If required arguments are missing or invalid.
            InvalidMethodError: If method is not GET or POST.
            InvalidPathError: If path is malformed.
            NetworkSafetyViolationError: If destination escapes local mock server.
            PayloadSizeLimitExceededError: If request body exceeds limit.
            ResponseSizeLimitExceededError: If response exceeds limit.
            HTTPExecutionError: If request fails to connect or times out.
        """
        if args is not None:
            if not isinstance(args, dict):
                raise InvalidArgumentError("Arguments must be provided as a dictionary.")
            params = dict(args)
            params.update(kwargs)
        else:
            params = kwargs

        if not params:
            raise InvalidArgumentError("Missing arguments for http_tool.")

        if "method" not in params:
            raise InvalidArgumentError("Missing required argument: 'method'.")

        raw_method = params["method"]
        if not isinstance(raw_method, str):
            raise InvalidArgumentError("Argument 'method' must be a string.")

        method = raw_method.strip().upper()
        if method not in self.SUPPORTED_METHODS:
            raise InvalidMethodError(
                f"Unsupported HTTP method: '{raw_method}'. Supported methods are: {', '.join(sorted(self.SUPPORTED_METHODS))}."
            )

        if "path" not in params:
            raise InvalidArgumentError("Missing required argument: 'path'.")

        path = self._validate_path(params["path"])

        # Construct target URL and verify authority containment
        target_url = f"{self.base_url}{path}"
        parsed_target = urllib.parse.urlsplit(target_url)
        parsed_base = urllib.parse.urlsplit(self.base_url)
        if (
            parsed_target.scheme != parsed_base.scheme
            or parsed_target.netloc != parsed_base.netloc
        ):
            raise NetworkSafetyViolationError(
                f"Target URL '{target_url}' escapes allowed base URL '{self.base_url}'."
            )

        # Handle headers
        headers: Dict[str, str] = {}
        raw_headers = params.get("headers")
        if raw_headers is not None:
            if not isinstance(raw_headers, dict):
                raise InvalidArgumentError("Headers must be a dictionary.")
            for k, v in raw_headers.items():
                headers[str(k)] = str(v)

        # Handle body
        body_bytes: Optional[bytes] = None
        if method == "POST":
            raw_body = params.get("body")
            if raw_body is not None:
                if isinstance(raw_body, str):
                    body_bytes = raw_body.encode("utf-8")
                else:
                    try:
                        body_bytes = json.dumps(raw_body).encode("utf-8")
                    except (TypeError, ValueError) as e:
                        raise InvalidArgumentError(f"Request body is not JSON-serializable: {e}") from e

                if len(body_bytes) > self.max_body_size:
                    raise PayloadSizeLimitExceededError(
                        f"Request body size ({len(body_bytes)} bytes) exceeds maximum allowed limit of {self.max_body_size} bytes."
                    )

                if "Content-Type" not in headers:
                    headers["Content-Type"] = "application/json"
        elif method == "GET":
            if params.get("body") is not None:
                raise InvalidArgumentError("GET requests must not contain a request body.")

        # Build and execute request
        req = urllib.request.Request(
            url=target_url,
            data=body_bytes,
            headers=headers,
            method=method,
        )

        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                status_code = resp.status
                resp_headers = dict(resp.headers)
                raw_resp = resp.read(self.max_response_size + 1)
        except urllib.error.HTTPError as e:
            status_code = e.code
            resp_headers = dict(e.headers)
            raw_resp = e.read(self.max_response_size + 1)
            e.close()
        except urllib.error.URLError as e:
            if isinstance(e.reason, TimeoutError):
                raise HTTPExecutionError(f"HTTP request timed out after {self.timeout}s: {e}") from e
            raise HTTPExecutionError(f"HTTP connection failed: {e}") from e
        except TimeoutError as e:
            raise HTTPExecutionError(f"HTTP request timed out after {self.timeout}s: {e}") from e

        if len(raw_resp) > self.max_response_size:
            raise ResponseSizeLimitExceededError(
                f"Response size exceeded maximum allowed limit of {self.max_response_size} bytes."
            )

        # Decode and parse response
        decoded = raw_resp.decode("utf-8", errors="replace")
        try:
            parsed_body = json.loads(decoded)
        except json.JSONDecodeError:
            parsed_body = decoded

        return {
            "status_code": status_code,
            "headers": resp_headers,
            "body": parsed_body,
            "success": 200 <= status_code < 400,
        }
