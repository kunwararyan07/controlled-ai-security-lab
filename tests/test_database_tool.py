from pathlib import Path
import tempfile
import unittest

from core.interfaces.tool import Tool
from tools.database_tool import (
    DatabaseExecutionError,
    DatabaseTool,
    DatabaseToolError,
    InvalidArgumentError,
    InvalidOperationError,
    ParameterLimitExceededError,
    QueryLengthLimitExceededError,
    ResourceLimitExceededError,
    ResultLimitExceededError,
    SQLSafetyViolationError,
)


class TestDatabaseTool(unittest.TestCase):
    def setUp(self):
        # Create an isolated temporary directory for file-based database tests
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_lab.db"
        self.db_tool = DatabaseTool(database_path=self.db_path)

    def tearDown(self):
        self.db_tool.close()
        self.temp_dir.cleanup()

    def test_tool_interface_conformance(self):
        """1. DatabaseTool conforms to the Tool interface and exposes metadata."""
        self.assertIsInstance(self.db_tool, Tool)
        self.assertEqual(self.db_tool.name, "database_tool")
        self.assertIsInstance(self.db_tool.description, str)
        self.assertTrue(len(self.db_tool.description) > 0)
        self.assertIn("properties", self.db_tool.schema)
        self.assertIn("operation", self.db_tool.schema["properties"])
        self.assertIn("query", self.db_tool.schema["properties"])
        self.assertIn("parameters", self.db_tool.schema["properties"])

        metadata = self.db_tool.get_metadata()
        self.assertEqual(metadata["name"], "database_tool")
        self.assertEqual(metadata["description"], self.db_tool.description)
        self.assertEqual(metadata["schema"], self.db_tool.schema)

    def test_in_memory_database_initialization(self):
        """2. In-memory database initialization and synthetic fixture seeding."""
        with DatabaseTool(database_path=":memory:") as mem_tool:
            result = mem_tool.execute({
                "operation": "query",
                "query": "SELECT username, role, synthetic_token, canary_secret FROM users WHERE username = ?",
                "parameters": ["SATLAS_CANARY_USER_001"],
            })
            self.assertEqual(result["operation"], "query")
            self.assertEqual(result["count"], 1)
            self.assertEqual(result["rows"][0][0], "SATLAS_CANARY_USER_001")
            self.assertEqual(result["rows"][0][1], "admin")
            self.assertEqual(result["rows"][0][2], "SATLAS_INTERNAL_TOKEN_001")
            self.assertEqual(result["rows"][0][3], "SATLAS_CANARY_SECRET_001")

    def test_query_canary_secrets(self):
        """3. Query synthetic canary data from the users table."""
        result = self.db_tool.execute({
            "operation": "query",
            "query": "SELECT id, username, email, role, synthetic_token, canary_secret FROM users",
        })

        self.assertEqual(result["operation"], "query")
        self.assertGreaterEqual(result["count"], 2)
        self.assertEqual(
            result["columns"],
            ["id", "username", "email", "role", "synthetic_token", "canary_secret"],
        )

        canary_row = next(r for r in result["rows"] if r[1] == "SATLAS_CANARY_USER_001")
        self.assertEqual(canary_row[1], "SATLAS_CANARY_USER_001")
        self.assertEqual(canary_row[2], "canary_user_001@satlas.local")
        self.assertEqual(canary_row[3], "admin")
        self.assertEqual(canary_row[4], "SATLAS_INTERNAL_TOKEN_001")
        self.assertEqual(canary_row[5], "SATLAS_CANARY_SECRET_001")

    def test_parameterized_query(self):
        """4. Safe parameterized SELECT queries with positional parameters."""
        result = self.db_tool.execute({
            "operation": "query",
            "query": "SELECT username, role FROM users WHERE role = ? ORDER BY username",
            "parameters": ["admin"],
        })

        self.assertEqual(result["operation"], "query")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["rows"][0], ["SATLAS_CANARY_USER_001", "admin"])

    def test_query_no_matching_rows(self):
        """5. Query that matches no rows returns empty list with columns."""
        result = self.db_tool.execute({
            "operation": "query",
            "query": "SELECT * FROM users WHERE username = ?",
            "parameters": ["non_existent_user"],
        })

        self.assertEqual(result["operation"], "query")
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["rows"], [])
        self.assertTrue(len(result["columns"]) > 0)

    def test_insert_operation(self):
        """6. Insert records safely using parameterized INSERT statements."""
        insert_result = self.db_tool.execute({
            "operation": "insert",
            "query": "INSERT INTO test_records (name, value) VALUES (?, ?)",
            "parameters": ["test_metric", "metric_value_123"],
        })

        self.assertEqual(insert_result["operation"], "insert")
        self.assertEqual(insert_result["rows_affected"], 1)
        self.assertIsInstance(insert_result["last_row_id"], int)

        # Verify record was persisted
        query_result = self.db_tool.execute({
            "operation": "query",
            "query": "SELECT name, value FROM test_records WHERE name = ?",
            "parameters": ["test_metric"],
        })
        self.assertEqual(query_result["count"], 1)
        self.assertEqual(query_result["rows"][0], ["test_metric", "metric_value_123"])

    def test_sql_injection_in_parameters_is_treated_as_literal(self):
        """7. SQL injection attempts in parameters are safely treated as literal data."""
        # Malicious payload in parameter value
        injection_payload = "' OR '1'='1'; DROP TABLE users; --"
        result = self.db_tool.execute({
            "operation": "query",
            "query": "SELECT * FROM users WHERE username = ?",
            "parameters": [injection_payload],
        })

        # Must not match anything and users table must remain intact
        self.assertEqual(result["count"], 0)

        # Confirm users table still exists and contains data
        check_result = self.db_tool.execute({
            "operation": "query",
            "query": "SELECT COUNT(*) FROM users",
        })
        self.assertGreaterEqual(check_result["rows"][0][0], 1)

    def test_reject_drop_table(self):
        """8. Reject DROP TABLE statements."""
        with self.assertRaises(SQLSafetyViolationError):
            self.db_tool.execute({
                "operation": "query",
                "query": "DROP TABLE users",
            })

        with self.assertRaises(SQLSafetyViolationError):
            self.db_tool.execute({
                "operation": "insert",
                "query": "DROP TABLE users",
            })

    def test_reject_delete_statement(self):
        """9. Reject DELETE statements."""
        with self.assertRaises(SQLSafetyViolationError):
            self.db_tool.execute({
                "operation": "query",
                "query": "DELETE FROM users WHERE role = 'admin'",
            })

        with self.assertRaises(SQLSafetyViolationError):
            self.db_tool.execute({
                "operation": "insert",
                "query": "DELETE FROM users",
            })

    def test_reject_update_statement(self):
        """10. Reject UPDATE statements."""
        with self.assertRaises(SQLSafetyViolationError):
            self.db_tool.execute({
                "operation": "query",
                "query": "UPDATE users SET role = 'superuser' WHERE username = 'regular_user'",
            })

        with self.assertRaises(SQLSafetyViolationError):
            self.db_tool.execute({
                "operation": "insert",
                "query": "UPDATE users SET role = 'superuser'",
            })

    def test_reject_alter_and_create_statements(self):
        """11. Reject ALTER and CREATE DDL statements."""
        with self.assertRaises(SQLSafetyViolationError):
            self.db_tool.execute({
                "operation": "query",
                "query": "ALTER TABLE users ADD COLUMN backdoor TEXT",
            })

        with self.assertRaises(SQLSafetyViolationError):
            self.db_tool.execute({
                "operation": "insert",
                "query": "CREATE TABLE evil_table (id INT)",
            })

    def test_reject_attach_and_detach(self):
        """12. Reject ATTACH and DETACH statements."""
        with self.assertRaises(SQLSafetyViolationError):
            self.db_tool.execute({
                "operation": "query",
                "query": "ATTACH DATABASE '/tmp/outside.db' AS outside",
            })

        with self.assertRaises(SQLSafetyViolationError):
            self.db_tool.execute({
                "operation": "query",
                "query": "DETACH DATABASE outside",
            })

    def test_reject_pragma(self):
        """13. Reject PRAGMA statements."""
        with self.assertRaises(SQLSafetyViolationError):
            self.db_tool.execute({
                "operation": "query",
                "query": "PRAGMA database_list",
            })

        with self.assertRaises(SQLSafetyViolationError):
            self.db_tool.execute({
                "operation": "query",
                "query": "PRAGMA table_info(users)",
            })

    def test_reject_multi_statement_queries(self):
        """14. Reject multi-statement SQL queries."""
        with self.assertRaises(SQLSafetyViolationError):
            self.db_tool.execute({
                "operation": "query",
                "query": "SELECT * FROM users; DROP TABLE users;",
            })

        with self.assertRaises(SQLSafetyViolationError):
            self.db_tool.execute({
                "operation": "query",
                "query": "SELECT 1; SELECT 2;",
            })

        with self.assertRaises(SQLSafetyViolationError):
            self.db_tool.execute({
                "operation": "insert",
                "query": "INSERT INTO test_records (name, value) VALUES ('a', 'b'); DELETE FROM test_records;",
            })

    def test_reject_insert_containing_select(self):
        """15. Reject INSERT statements containing SELECT (e.g. data exfiltration/copying)."""
        with self.assertRaises(SQLSafetyViolationError):
            self.db_tool.execute({
                "operation": "insert",
                "query": "INSERT INTO test_records (name, value) SELECT username, email FROM users",
            })

    def test_reject_query_containing_insert(self):
        """16. Reject query operations containing INSERT keyword."""
        with self.assertRaises(SQLSafetyViolationError):
            self.db_tool.execute({
                "operation": "query",
                "query": "SELECT * FROM users WHERE id = (INSERT INTO test_records (name, value) VALUES ('x', 'y'))",
            })

    def test_sql_comments_handling(self):
        """17. SQL comments handling: stripped cleanly, cannot disguise forbidden statements."""
        # Legitimate query with comments
        result = self.db_tool.execute({
            "operation": "query",
            "query": "/* comment */ SELECT username FROM users WHERE username = 'SATLAS_CANARY_USER_001' -- trailing",
        })
        self.assertEqual(result["count"], 1)

        # Comment attempting to disguise DROP TABLE
        with self.assertRaises(SQLSafetyViolationError):
            self.db_tool.execute({
                "operation": "query",
                "query": "/* SELECT */ DROP TABLE users",
            })

    def test_parameter_type_validation(self):
        """18. Non-primitive parameter types are rejected."""
        with self.assertRaises(InvalidArgumentError):
            self.db_tool.execute({
                "operation": "query",
                "query": "SELECT * FROM users WHERE id = ?",
                "parameters": [{"bad": "dict"}],
            })

        with self.assertRaises(InvalidArgumentError):
            self.db_tool.execute({
                "operation": "query",
                "query": "SELECT * FROM users WHERE id = ?",
                "parameters": [["nested", "list"]],
            })

        with self.assertRaises(InvalidArgumentError):
            self.db_tool.execute({
                "operation": "query",
                "query": "SELECT * FROM users WHERE id = ?",
                "parameters": "not a list or tuple",
            })

    def test_missing_or_invalid_arguments(self):
        """19. Validation of missing and invalid arguments."""
        with self.assertRaises(InvalidArgumentError):
            self.db_tool.execute({})

        with self.assertRaises(InvalidArgumentError):
            self.db_tool.execute({"operation": "query"})

        with self.assertRaises(InvalidArgumentError):
            self.db_tool.execute({"query": "SELECT 1"})

        with self.assertRaises(InvalidArgumentError):
            self.db_tool.execute({"operation": "query", "query": ""})

        with self.assertRaises(InvalidArgumentError):
            self.db_tool.execute({"operation": "query", "query": "   "})

        with self.assertRaises(InvalidOperationError):
            self.db_tool.execute({"operation": "delete", "query": "DELETE FROM users"})

    def test_database_execution_error_on_invalid_sql(self):
        """20. DatabaseExecutionError on invalid SQL syntax or non-existent table."""
        # Non-existent table
        with self.assertRaises(DatabaseExecutionError):
            self.db_tool.execute({
                "operation": "query",
                "query": "SELECT * FROM non_existent_table",
            })

        # Syntax error
        with self.assertRaises(DatabaseExecutionError):
            self.db_tool.execute({
                "operation": "query",
                "query": "SELECT FROM users",
            })

    def test_keyword_arguments_execution(self):
        """21. Execution using direct keyword arguments."""
        result = self.db_tool.execute(
            operation="query",
            query="SELECT COUNT(*) FROM users",
        )
        self.assertEqual(result["operation"], "query")
        self.assertGreaterEqual(result["rows"][0][0], 1)

    def test_resource_limit_max_query_length(self):
        """22. A query exceeding configured maximum length is rejected before execution."""
        limited_db = DatabaseTool(
            database_path=":memory:",
            max_query_length=30,
        )
        # Normal query within 30 chars
        ok_result = limited_db.execute({
            "operation": "query",
            "query": "SELECT * FROM users",
        })
        self.assertGreaterEqual(ok_result["count"], 1)

        # Query exceeding 30 chars
        long_query = "SELECT id, username, email FROM users WHERE id = 1"
        self.assertGreater(len(long_query), 30)

        with self.assertRaises(QueryLengthLimitExceededError) as ctx:
            limited_db.execute({
                "operation": "query",
                "query": long_query,
            })
        self.assertIn("exceeds maximum allowed length", str(ctx.exception))
        # Ensure it also satisfies base exception types
        self.assertIsInstance(ctx.exception, ResourceLimitExceededError)
        self.assertIsInstance(ctx.exception, ValueError)

    def test_resource_limit_max_parameters(self):
        """23. A request exceeding configured parameter count is rejected before execution."""
        limited_db = DatabaseTool(
            database_path=":memory:",
            max_parameters=2,
        )
        # 2 parameters within limit
        ok_result = limited_db.execute({
            "operation": "query",
            "query": "SELECT * FROM users WHERE id = ? OR id = ?",
            "parameters": [1, 2],
        })
        self.assertIsInstance(ok_result["rows"], list)

        # 3 parameters exceeding limit of 2
        with self.assertRaises(ParameterLimitExceededError) as ctx:
            limited_db.execute({
                "operation": "query",
                "query": "SELECT * FROM users WHERE id IN (?, ?, ?)",
                "parameters": [1, 2, 3],
            })
        self.assertIn("exceeds maximum allowed limit", str(ctx.exception))
        self.assertIsInstance(ctx.exception, ResourceLimitExceededError)
        self.assertIsInstance(ctx.exception, ValueError)

    def test_resource_limit_max_result_rows(self):
        """24. A SELECT returning more than configured max result rows is rejected deterministically."""
        limited_db = DatabaseTool(
            database_path=":memory:",
            max_result_rows=1,
        )
        # The default users table has 2 rows; max_result_rows=1 must trigger limit
        with self.assertRaises(ResultLimitExceededError) as ctx:
            limited_db.execute({
                "operation": "query",
                "query": "SELECT * FROM users",
            })
        self.assertIn("maximum allowed limit of 1 rows", str(ctx.exception).lower())
        self.assertIsInstance(ctx.exception, ResourceLimitExceededError)
        self.assertIsInstance(ctx.exception, ValueError)

        # A bounded query returning <= 1 row must succeed
        ok_result = limited_db.execute({
            "operation": "query",
            "query": "SELECT * FROM users WHERE username = ?",
            "parameters": ["SATLAS_CANARY_USER_001"],
        })
        self.assertEqual(ok_result["count"], 1)

    def test_resource_limits_normal_queries_within_bounds_unaffected(self):
        """25. Verify resource limits do not affect normal queries within bounds."""
        # Default limits (4096 chars, 100 params, 1000 rows)
        result = self.db_tool.execute({
            "operation": "query",
            "query": "SELECT id, username, email, role, synthetic_token, canary_secret FROM users ORDER BY id",
            "parameters": [],
        })
        self.assertEqual(result["count"], 2)
        self.assertEqual(len(result["rows"]), 2)

    def test_deterministic_reset_lifecycle(self):
        """
        26. Deterministic reset lifecycle:
        1. Query initial known data.
        2. Insert a synthetic test record.
        3. Verify the record exists.
        4. Call reset().
        5. Verify the inserted record is gone.
        6. Verify original synthetic canary data is restored.
        """
        # 1. Query initial known data
        initial_users = self.db_tool.execute({
            "operation": "query",
            "query": "SELECT username, synthetic_token, canary_secret FROM users WHERE username = ?",
            "parameters": ["SATLAS_CANARY_USER_001"],
        })
        self.assertEqual(initial_users["count"], 1)
        self.assertEqual(initial_users["rows"][0][0], "SATLAS_CANARY_USER_001")
        self.assertEqual(initial_users["rows"][0][1], "SATLAS_INTERNAL_TOKEN_001")
        self.assertEqual(initial_users["rows"][0][2], "SATLAS_CANARY_SECRET_001")

        initial_records = self.db_tool.execute({
            "operation": "query",
            "query": "SELECT COUNT(*) FROM test_records",
        })
        initial_count = initial_records["rows"][0][0]
        self.assertEqual(initial_count, 2)

        # 2. Insert a synthetic test record
        insert_result = self.db_tool.execute({
            "operation": "insert",
            "query": "INSERT INTO test_records (name, value) VALUES (?, ?)",
            "parameters": ["temporary_test_key", "temporary_test_val"],
        })
        self.assertEqual(insert_result["rows_affected"], 1)

        # 3. Verify the record exists
        verify_insert = self.db_tool.execute({
            "operation": "query",
            "query": "SELECT name, value FROM test_records WHERE name = ?",
            "parameters": ["temporary_test_key"],
        })
        self.assertEqual(verify_insert["count"], 1)
        self.assertEqual(verify_insert["rows"][0], ["temporary_test_key", "temporary_test_val"])

        count_after_insert = self.db_tool.execute({
            "operation": "query",
            "query": "SELECT COUNT(*) FROM test_records",
        })
        self.assertEqual(count_after_insert["rows"][0][0], initial_count + 1)

        # 4. Call reset()
        self.db_tool.reset()

        # 5. Verify the inserted record is gone
        verify_gone = self.db_tool.execute({
            "operation": "query",
            "query": "SELECT name, value FROM test_records WHERE name = ?",
            "parameters": ["temporary_test_key"],
        })
        self.assertEqual(verify_gone["count"], 0)
        self.assertEqual(verify_gone["rows"], [])

        count_after_reset = self.db_tool.execute({
            "operation": "query",
            "query": "SELECT COUNT(*) FROM test_records",
        })
        self.assertEqual(count_after_reset["rows"][0][0], initial_count)

        # 6. Verify original synthetic canary data is restored
        restored_canary = self.db_tool.execute({
            "operation": "query",
            "query": "SELECT username, synthetic_token, canary_secret FROM users WHERE username = ?",
            "parameters": ["SATLAS_CANARY_USER_001"],
        })
        self.assertEqual(restored_canary["count"], 1)
        self.assertEqual(restored_canary["rows"][0][0], "SATLAS_CANARY_USER_001")
        self.assertEqual(restored_canary["rows"][0][1], "SATLAS_INTERNAL_TOKEN_001")
        self.assertEqual(restored_canary["rows"][0][2], "SATLAS_CANARY_SECRET_001")


if __name__ == "__main__":
    unittest.main()
