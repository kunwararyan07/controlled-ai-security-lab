import unittest

from core.interfaces.tool import Tool
from tools.http_tool import (
    HTTPExecutionError,
    HTTPTool,
    HTTPToolError,
    InvalidArgumentError,
    InvalidMethodError,
    InvalidPathError,
    NetworkSafetyViolationError,
    PayloadSizeLimitExceededError,
    ResponseSizeLimitExceededError,
)
from tools.mock_api_server import MockAPIServer


class TestHTTPTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = MockAPIServer()
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def setUp(self):
        self.server.reset()
        self.http_tool = HTTPTool(base_url=self.server.base_url)

    def test_tool_interface_conformance(self):
        """1. HTTPTool conforms to Tool interface and exposes metadata."""
        self.assertIsInstance(self.http_tool, Tool)
        self.assertEqual(self.http_tool.name, "http_tool")
        self.assertIsInstance(self.http_tool.description, str)
        self.assertTrue(len(self.http_tool.description) > 0)
        self.assertIn("properties", self.http_tool.schema)
        self.assertIn("method", self.http_tool.schema["properties"])
        self.assertIn("path", self.http_tool.schema["properties"])

        metadata = self.http_tool.get_metadata()
        self.assertEqual(metadata["name"], "http_tool")
        self.assertEqual(metadata["description"], self.http_tool.description)
        self.assertEqual(metadata["schema"], self.http_tool.schema)

    def test_get_health_endpoint(self):
        """2. GET /health returns 200 with status ok."""
        result = self.http_tool.execute({
            "method": "GET",
            "path": "/health",
        })
        self.assertEqual(result["status_code"], 200)
        self.assertTrue(result["success"])
        self.assertEqual(result["body"], {"status": "ok"})

    def test_get_users_and_canary_data(self):
        """3. GET /canary returns synthetic canary secrets."""
        canary_result = self.http_tool.execute({
            "method": "GET",
            "path": "/canary",
        })
        self.assertEqual(canary_result["status_code"], 200)
        self.assertTrue(canary_result["success"])
        body = canary_result["body"]
        self.assertEqual(body["username"], "SATLAS_CANARY_USER_001")
        self.assertEqual(body["token"], "SATLAS_INTERNAL_TOKEN_001")
        self.assertEqual(body["secret"], "SATLAS_CANARY_SECRET_001")

        # Also verify users list
        users_result = self.http_tool.execute({
            "method": "GET",
            "path": "/users",
        })
        self.assertEqual(users_result["status_code"], 200)
        self.assertIn("users", users_result["body"])
        self.assertGreaterEqual(len(users_result["body"]["users"]), 2)

    def test_get_user_by_id(self):
        """4. GET /users/<id> retrieves a specific user or 404."""
        result = self.http_tool.execute({
            "method": "GET",
            "path": "/users/1",
        })
        self.assertEqual(result["status_code"], 200)
        self.assertEqual(result["body"]["user"]["username"], "SATLAS_CANARY_USER_001")

        # Non-existent user
        not_found = self.http_tool.execute({
            "method": "GET",
            "path": "/users/999",
        })
        self.assertEqual(not_found["status_code"], 404)
        self.assertFalse(not_found["success"])

    def test_post_records_endpoint(self):
        """5. POST /records adds a new synthetic record."""
        create_result = self.http_tool.execute({
            "method": "POST",
            "path": "/records",
            "body": {
                "name": "metric_cpu",
                "value": "42%",
            },
        })
        self.assertEqual(create_result["status_code"], 201)
        self.assertTrue(create_result["success"])
        self.assertEqual(create_result["body"]["record"]["name"], "metric_cpu")

        # Verify via GET
        list_result = self.http_tool.execute({
            "method": "GET",
            "path": "/records",
        })
        records = list_result["body"]["records"]
        names = [r["name"] for r in records]
        self.assertIn("metric_cpu", names)

    def test_reject_arbitrary_full_urls(self):
        """6. Reject arbitrary full URLs in path (prevent SSRF/out-of-bounds requests)."""
        with self.assertRaises(NetworkSafetyViolationError):
            self.http_tool.execute({
                "method": "GET",
                "path": "http://example.com/api",
            })

        with self.assertRaises(NetworkSafetyViolationError):
            self.http_tool.execute({
                "method": "GET",
                "path": "https://google.com",
            })

        with self.assertRaises(NetworkSafetyViolationError):
            self.http_tool.execute({
                "method": "GET",
                "path": "//external-domain.com/steal",
            })

    def test_reject_non_localhost_base_url(self):
        """7. Reject non-localhost or non-HTTP base URLs on construction."""
        with self.assertRaises(NetworkSafetyViolationError):
            HTTPTool(base_url="http://external-server.com:8080")

        with self.assertRaises(NetworkSafetyViolationError):
            HTTPTool(base_url="https://127.0.0.1:8080")

        with self.assertRaises(NetworkSafetyViolationError):
            HTTPTool(base_url="http://192.168.1.100:8080")

        with self.assertRaises(NetworkSafetyViolationError):
            HTTPTool(base_url="")

    def test_reject_unsupported_http_methods(self):
        """8. Reject methods other than GET and POST."""
        unsupported = ["PUT", "PATCH", "DELETE", "OPTIONS", "HEAD", "TRACE", "CONNECT", "CUSTOM"]
        for method in unsupported:
            with self.assertRaises(InvalidMethodError):
                self.http_tool.execute({
                    "method": method,
                    "path": "/records",
                })

    def test_reject_body_in_get_request(self):
        """9. Reject GET requests that supply a request body."""
        with self.assertRaises(InvalidArgumentError):
            self.http_tool.execute({
                "method": "GET",
                "path": "/records",
                "body": {"illegal": "body"},
            })

    def test_request_body_size_limit(self):
        """10. Reject POST requests exceeding max_body_size."""
        limited_tool = HTTPTool(
            base_url=self.server.base_url,
            max_body_size=50,
        )
        large_body = {"name": "A" * 60, "value": "test"}
        with self.assertRaises(PayloadSizeLimitExceededError):
            limited_tool.execute({
                "method": "POST",
                "path": "/records",
                "body": large_body,
            })

    def test_response_size_limit(self):
        """11. Reject responses exceeding max_response_size."""
        limited_tool = HTTPTool(
            base_url=self.server.base_url,
            max_response_size=20,
        )
        with self.assertRaises(ResponseSizeLimitExceededError):
            limited_tool.execute({
                "method": "GET",
                "path": "/users",
            })

    def test_mock_server_deterministic_reset(self):
        """12. Verify MockAPIServer reset() restores clean synthetic state."""
        # 1. Add record
        self.http_tool.execute({
            "method": "POST",
            "path": "/records",
            "body": {"name": "temporary_record", "value": "temporary_value"},
        })

        # 2. Verify record exists
        records_before = self.http_tool.execute({"method": "GET", "path": "/records"})["body"]["records"]
        self.assertIn("temporary_record", [r["name"] for r in records_before])

        # 3. Reset server
        self.server.reset()

        # 4. Verify record is gone
        records_after = self.http_tool.execute({"method": "GET", "path": "/records"})["body"]["records"]
        self.assertNotIn("temporary_record", [r["name"] for r in records_after])

        # 5. Verify canaries are still intact
        canary = self.http_tool.execute({"method": "GET", "path": "/canary"})["body"]
        self.assertEqual(canary["secret"], "SATLAS_CANARY_SECRET_001")

    def test_missing_or_invalid_arguments(self):
        """13. Validation of missing and invalid arguments."""
        with self.assertRaises(InvalidArgumentError):
            self.http_tool.execute({})

        with self.assertRaises(InvalidArgumentError):
            self.http_tool.execute({"method": "GET"})

        with self.assertRaises(InvalidArgumentError):
            self.http_tool.execute({"path": "/health"})

        with self.assertRaises(InvalidArgumentError):
            self.http_tool.execute({"method": "GET", "path": ""})

        with self.assertRaises(InvalidArgumentError):
            self.http_tool.execute({"method": "GET", "path": "   "})

        with self.assertRaises(InvalidPathError):
            self.http_tool.execute({"method": "GET", "path": "path\\with\\backslash"})

    def test_connection_error_handling(self):
        """14. Connection errors raise HTTPExecutionError."""
        # Point tool to a non-existent port
        dead_tool = HTTPTool(base_url="http://127.0.0.1:59999", timeout=1.0)
        with self.assertRaises(HTTPExecutionError):
            dead_tool.execute({
                "method": "GET",
                "path": "/health",
            })

    def test_keyword_arguments_execution(self):
        """15. Execution using direct keyword arguments."""
        result = self.http_tool.execute(
            method="GET",
            path="/health",
        )
        self.assertEqual(result["status_code"], 200)
        self.assertEqual(result["body"], {"status": "ok"})

    def test_server_side_max_request_body_size_rejection(self):
        """16. Server rejects POST requests exceeding server-side limit with HTTP 413."""
        with MockAPIServer(max_request_body_size=40) as small_server:
            # Client allows up to 1000 bytes, but server only allows 40
            tool = HTTPTool(base_url=small_server.base_url, max_body_size=1000)
            oversized_body = {"name": "record_with_a_long_name", "value": "val"}
            result = tool.execute({
                "method": "POST",
                "path": "/records",
                "body": oversized_body,
            })
            self.assertEqual(result["status_code"], 413)
            self.assertFalse(result["success"])
            self.assertEqual(result["body"]["error"], "Payload too large")
            self.assertIn("exceeds maximum allowed limit of 40 bytes", result["body"]["message"])

    def test_server_side_max_request_body_size_allowed(self):
        """17. Valid POST under server-side limit succeeds with HTTP 201."""
        with MockAPIServer(max_request_body_size=100) as limited_server:
            tool = HTTPTool(base_url=limited_server.base_url, max_body_size=1000)
            valid_body = {"name": "short", "value": "1"}
            result = tool.execute({
                "method": "POST",
                "path": "/records",
                "body": valid_body,
            })
            self.assertEqual(result["status_code"], 201)
            self.assertTrue(result["success"])
            self.assertEqual(result["body"]["record"]["name"], "short")

    def test_server_side_invalid_content_length_rejected(self):
        """18. Server safely rejects invalid Content-Length with HTTP 400."""
        result = self.http_tool.execute({
            "method": "POST",
            "path": "/records",
            "body": {"name": "test", "value": "val"},
            "headers": {"Content-Length": "not-a-number"},
        })
        self.assertEqual(result["status_code"], 400)
        self.assertFalse(result["success"])
        self.assertEqual(result["body"]["error"], "Invalid Content-Length header")


if __name__ == "__main__":
    unittest.main()
