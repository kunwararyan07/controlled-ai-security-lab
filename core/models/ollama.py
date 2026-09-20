import json
import re
import socket
from typing import Any, Dict, Optional
import urllib.error
import urllib.parse
import urllib.request

from core.models.base import ModelAdapter


class OllamaError(Exception):
    """Base exception for all Ollama adapter errors."""
    pass


class OllamaConfigurationError(OllamaError, ValueError):
    """Raised when OllamaAdapter configuration is invalid."""
    pass


class OllamaConnectionError(OllamaError, RuntimeError):
    """Raised when connection to Ollama server fails."""
    pass


class OllamaTimeoutError(OllamaConnectionError, TimeoutError):
    """Raised when a request to Ollama times out."""
    pass


class OllamaResponseError(OllamaError, RuntimeError):
    """Raised when Ollama returns an error status, malformed JSON, or missing fields."""
    pass


def _validate_model_name(model_name: str) -> str:
    """
    Validate configured model name.

    Rejects:
    - Non-string values
    - Empty or whitespace-only names
    - Values containing control characters (ASCII < 32 or 127)
    - Values containing characters outside the safe Ollama identifier set [a-zA-Z0-9_.:/-]
    """
    if not isinstance(model_name, str):
        raise OllamaConfigurationError("Model name must be a string.")
    name = model_name.strip()
    if not name:
        raise OllamaConfigurationError("Model name cannot be empty or whitespace.")

    for ch in name:
        if ord(ch) < 32 or ord(ch) == 127:
            raise OllamaConfigurationError(
                f"Model name contains invalid control character: {repr(ch)}"
            )

    if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9_.:/-]*$", name):
        raise OllamaConfigurationError(
            f"Invalid model name format: '{name}'. Must start with alphanumeric and contain only [a-zA-Z0-9_.:/-]."
        )

    return name


def _validate_base_url(base_url: str) -> str:
    """
    Validate that configured base URL is safe and confined to localhost/loopback.

    Enforces:
    - Non-empty string
    - 'http' scheme only
    - No credentials (username or password)
    - Hostname must resolve to loopback (127.0.0.1, localhost, ::1, or 127.x.x.x)
    - Valid port number (1-65535) if specified
    - No unexpected path (must be empty or '/')
    - No query parameters or fragments
    """
    if not isinstance(base_url, str):
        raise OllamaConfigurationError("Base URL must be a string.")
    url = base_url.strip()
    if not url:
        raise OllamaConfigurationError("Base URL cannot be empty or whitespace.")

    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "http":
        raise OllamaConfigurationError(
            f"Ollama base URL must use 'http' scheme, got '{parsed.scheme}'."
        )

    if parsed.username or parsed.password or "@" in parsed.netloc:
        raise OllamaConfigurationError(
            "Ollama base URL must not contain credentials (username or password)."
        )

    hostname = parsed.hostname
    if not hostname:
        raise OllamaConfigurationError("Ollama base URL must include a hostname.")

    valid_loopbacks = {"127.0.0.1", "localhost", "::1"}
    is_loopback = False
    if hostname in valid_loopbacks:
        is_loopback = True
    elif hostname.startswith("127."):
        parts = hostname.split(".")
        if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
            is_loopback = True

    if not is_loopback:
        raise OllamaConfigurationError(
            f"Ollama base URL must point to localhost/loopback, got '{hostname}'."
        )

    if parsed.port is not None:
        if not (1 <= parsed.port <= 65535):
            raise OllamaConfigurationError(
                f"Ollama base URL port out of range: {parsed.port}."
            )

    if parsed.path not in ("", "/"):
        raise OllamaConfigurationError(
            f"Ollama base URL must not contain a path, got '{parsed.path}'."
        )

    if parsed.query or parsed.fragment:
        raise OllamaConfigurationError(
            "Ollama base URL must not contain query parameters or fragments."
        )

    return f"http://{parsed.netloc}".rstrip("/")


class OllamaAdapter(ModelAdapter):
    """
    Adapter for communicating with a local Ollama model via Ollama's HTTP API.

    Implements the ModelAdapter interface, sending prompts to /api/generate and
    returning raw text responses. Network communication is strictly confined to
    localhost/loopback.
    """

    DEFAULT_BASE_URL = "http://127.0.0.1:11434"
    DEFAULT_TIMEOUT = 30.0
    DEFAULT_MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10 MB

    def __init__(
        self,
        model_name: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        options: Optional[Dict[str, Any]] = None,
        max_response_size: int = DEFAULT_MAX_RESPONSE_SIZE,
    ) -> None:
        """
        Initialize the Ollama adapter.

        Args:
            model_name: The name of the local Ollama model (e.g., 'qwen3:4b', 'mistral').
            base_url: The local Ollama base URL (defaults to 'http://127.0.0.1:11434').
            timeout: Request timeout in seconds.
            options: Optional dictionary of Ollama generation options.
            max_response_size: Maximum allowed response size in bytes.
        """
        self.model_name = _validate_model_name(model_name)
        self.base_url = _validate_base_url(base_url)
        self.generate_url = f"{self.base_url}/api/generate"

        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise OllamaConfigurationError("timeout must be a positive number.")
        self.timeout = float(timeout)

        if options is not None:
            if not isinstance(options, dict):
                raise OllamaConfigurationError("options must be a dictionary if provided.")
            self.options = dict(options)
        else:
            self.options = None

        if not isinstance(max_response_size, int) or max_response_size <= 0:
            raise OllamaConfigurationError("max_response_size must be a positive integer.")
        self.max_response_size = max_response_size

    def generate(self, prompt: str) -> str:
        """
        Generate a response from the Ollama model for the given prompt.

        Args:
            prompt: Input text prompt.

        Returns:
            The raw text response from the model.

        Raises:
            TypeError: If prompt is not a string.
            OllamaTimeoutError: If the request to Ollama times out.
            OllamaConnectionError: If connecting to Ollama fails.
            OllamaResponseError: If Ollama returns an error status, malformed JSON, or missing fields.
        """
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        if self.options is not None:
            payload["options"] = self.options

        try:
            req_data = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError) as e:
            raise OllamaConfigurationError(f"Failed to serialize request payload: {e}") from e

        req = urllib.request.Request(
            self.generate_url,
            data=req_data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_bytes = resp.read(self.max_response_size + 1)
                if len(resp_bytes) > self.max_response_size:
                    raise OllamaResponseError(
                        f"Response size exceeded limit of {self.max_response_size} bytes."
                    )
                resp_text = resp_bytes.decode("utf-8", errors="replace")

        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read(4096).decode("utf-8", errors="replace")
            except Exception:
                pass
            raise OllamaResponseError(
                f"Ollama HTTP error {e.code}: {e.reason}. Details: {err_body}"
            ) from e

        except urllib.error.URLError as e:
            reason = e.reason
            if isinstance(reason, (socket.timeout, TimeoutError)):
                raise OllamaTimeoutError(
                    f"Request to Ollama timed out after {self.timeout}s."
                ) from e
            raise OllamaConnectionError(
                f"Failed to connect to Ollama at '{self.generate_url}': {reason}"
            ) from e

        except (socket.timeout, TimeoutError) as e:
            raise OllamaTimeoutError(
                f"Request to Ollama timed out after {self.timeout}s."
            ) from e

        except Exception as e:
            if isinstance(e, OllamaError):
                raise
            raise OllamaConnectionError(
                f"Unexpected error communicating with Ollama: {e}"
            ) from e

        try:
            data = json.loads(resp_text)
        except json.JSONDecodeError as e:
            raise OllamaResponseError(
                f"Malformed JSON response from Ollama: {e}. Raw response: {resp_text[:200]}"
            ) from e

        if not isinstance(data, dict):
            raise OllamaResponseError(
                f"Expected JSON object from Ollama, got {type(data).__name__}."
            )

        if "error" in data and "response" not in data:
            raise OllamaResponseError(f"Ollama returned error: {data['error']}")

        if "response" not in data:
            raise OllamaResponseError(
                "Ollama response missing required 'response' field."
            )

        if not isinstance(data["response"], str):
            raise OllamaResponseError(
                f"Ollama 'response' field must be a string, got {type(data['response']).__name__}."
            )

        return data["response"]
