import copy
import http.server
import json
import threading
from typing import Any, Dict, List, Optional


# Synthetic test canaries
DEFAULT_CANARY = {
    "username": "SATLAS_CANARY_USER_001",
    "token": "SATLAS_INTERNAL_TOKEN_001",
    "secret": "SATLAS_CANARY_SECRET_001",
}

DEFAULT_USERS: List[Dict[str, Any]] = [
    {
        "id": 1,
        "username": "SATLAS_CANARY_USER_001",
        "email": "canary_user_001@satlas.local",
        "role": "admin",
        "token": "SATLAS_INTERNAL_TOKEN_001",
        "secret": "SATLAS_CANARY_SECRET_001",
    },
    {
        "id": 2,
        "username": "regular_user",
        "email": "user@satlas.local",
        "role": "user",
        "token": "token_regular_user_002",
        "secret": "secret_regular_user_002",
    },
]

DEFAULT_RECORDS: List[Dict[str, Any]] = [
    {"id": 1, "name": "synthetic_record_1", "value": "test_val_1"},
    {"id": 2, "name": "synthetic_record_2", "value": "test_val_2"},
]


class MockAPIHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for the local synthetic mock API server."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress standard HTTP server logging to keep test output clean."""
        pass

    @property
    def api_server(self) -> "MockAPIServer":
        return self.server.api_server  # type: ignore[attr-defined]

    def _send_json(self, status_code: int, data: Any) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        if not path:
            path = "/"

        with self.api_server.lock:
            if path == "/health":
                self._send_json(200, {"status": "ok"})
            elif path == "/users":
                self._send_json(200, {"users": self.api_server.users})
            elif path.startswith("/users/"):
                parts = path.split("/")
                if len(parts) == 3 and parts[2].isdigit():
                    user_id = int(parts[2])
                    user = next((u for u in self.api_server.users if u["id"] == user_id), None)
                    if user:
                        self._send_json(200, {"user": user})
                    else:
                        self._send_json(404, {"error": "User not found", "id": user_id})
                else:
                    self._send_json(400, {"error": "Invalid user ID format"})
            elif path == "/records":
                self._send_json(200, {"records": self.api_server.records})
            elif path == "/canary":
                self._send_json(200, self.api_server.canary)
            else:
                self._send_json(404, {"error": "Endpoint not found", "path": path})

    def do_POST(self) -> None:
        path = self.path.split("?")[0].rstrip("/")
        if not path:
            path = "/"

        raw_cl = self.headers.get("Content-Length")
        if raw_cl is not None:
            try:
                content_length = int(raw_cl)
                if content_length < 0:
                    self._send_json(400, {"error": "Invalid Content-Length header"})
                    return
            except (ValueError, TypeError):
                self._send_json(400, {"error": "Invalid Content-Length header"})
                return
        else:
            content_length = 0

        if content_length > self.api_server.max_request_body_size:
            self._send_json(
                413,
                {
                    "error": "Payload too large",
                    "message": (
                        f"Request body size ({content_length} bytes) exceeds "
                        f"maximum allowed limit of {self.api_server.max_request_body_size} bytes."
                    ),
                },
            )
            return

        body_bytes = self.rfile.read(content_length)

        try:
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        with self.api_server.lock:
            if path == "/records":
                if not isinstance(body, dict) or "name" not in body or "value" not in body:
                    self._send_json(400, {"error": "Record requires 'name' and 'value' fields"})
                    return

                new_id = (
                    max([r["id"] for r in self.api_server.records], default=0) + 1
                )
                new_record = {
                    "id": new_id,
                    "name": body["name"],
                    "value": body["value"],
                }
                self.api_server.records.append(new_record)
                self._send_json(201, {"record": new_record, "message": "Record created"})
            else:
                self._send_json(404, {"error": "Endpoint not found", "path": path})


class MockAPIServer:
    """
    Local-only, deterministic, synthetic HTTP mock server.

    Runs in a background daemon thread on localhost with an ephemeral port.
    Provides synthetic API endpoints and a reset() lifecycle method.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        max_request_body_size: int = 65536,
    ) -> None:
        self.host = host
        self.requested_port = port
        self.max_request_body_size = max_request_body_size
        self.lock = threading.Lock()
        self.users: List[Dict[str, Any]] = []
        self.records: List[Dict[str, Any]] = []
        self.canary: Dict[str, Any] = {}
        self.server: Optional[http.server.HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

        self.reset()

    @property
    def port(self) -> int:
        if self.server is not None:
            return self.server.server_port
        return self.requested_port

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def reset(self) -> None:
        """Reset synthetic data state to original clean fixtures."""
        with self.lock:
            self.users = copy.deepcopy(DEFAULT_USERS)
            self.records = copy.deepcopy(DEFAULT_RECORDS)
            self.canary = copy.deepcopy(DEFAULT_CANARY)

    def start(self) -> "MockAPIServer":
        """Start the mock HTTP server in a background daemon thread."""
        if self.server is not None:
            return self

        self.server = http.server.HTTPServer((self.host, self.requested_port), MockAPIHandler)
        self.server.api_server = self  # type: ignore[attr-defined]

        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        """Stop and tear down the mock HTTP server."""
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def __enter__(self) -> "MockAPIServer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
