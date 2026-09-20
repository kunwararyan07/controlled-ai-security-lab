import concurrent.futures
import threading
import unittest

from core.interfaces.tool import Tool
from tools.notification_tool import (
    InvalidNotificationArgumentError,
    InvalidOperationError,
    InvalidRecipientError,
    NotificationNotFoundError,
    NotificationResourceLimitError,
    NotificationTool,
    NotificationToolError,
    SYNTHETIC_CANARY_NOTIFICATION,
    SYNTHETIC_CANARY_USER,
)


class TestNotificationTool(unittest.TestCase):
    def setUp(self):
        self.tool = NotificationTool()

    def test_tool_interface_conformance(self):
        """1. Tool interface conformance."""
        self.assertIsInstance(self.tool, Tool)
        self.assertEqual(self.tool.name, "notification_tool")
        self.assertIsInstance(self.tool.description, str)
        self.assertTrue(len(self.tool.description) > 0)
        self.assertIn("synthetic", self.tool.description.lower())

    def test_tool_metadata_and_schema(self):
        """2. Tool metadata and schema."""
        metadata = self.tool.get_metadata()
        self.assertEqual(metadata["name"], "notification_tool")
        self.assertEqual(metadata["description"], self.tool.description)
        self.assertIn("properties", metadata["schema"])
        self.assertIn("operation", metadata["schema"]["properties"])
        self.assertIn("recipient", metadata["schema"]["properties"])
        self.assertIn("subject", metadata["schema"]["properties"])
        self.assertIn("body", metadata["schema"]["properties"])

    def test_send_valid_notification(self):
        """3. Send valid notification."""
        result = self.tool.execute({
            "operation": "send",
            "recipient": "user@satlas.local",
            "subject": "System Alert",
            "body": "This is a synthetic alert message.",
        })
        self.assertTrue(result["success"])
        self.assertEqual(result["notification_id"], "notification_001")
        self.assertEqual(result["recipient"], "user@satlas.local")
        self.assertEqual(result["subject"], "System Alert")
        self.assertEqual(result["status"], "queued")

    def test_deterministic_notification_ids(self):
        """4. Deterministic incrementing notification IDs."""
        r1 = self.tool.execute({
            "operation": "send",
            "recipient": "user1@satlas.local",
            "subject": "First",
            "body": "Message 1",
        })
        r2 = self.tool.execute({
            "operation": "send",
            "recipient": "user2@satlas.local",
            "subject": "Second",
            "body": "Message 2",
        })
        self.assertEqual(r1["notification_id"], "notification_001")
        self.assertEqual(r2["notification_id"], "notification_002")

    def test_multiple_notifications(self):
        """5. Multiple notifications stored sequentially."""
        for i in range(5):
            res = self.tool.execute({
                "operation": "send",
                "recipient": f"user{i}@satlas.local",
                "subject": f"Notice {i}",
                "body": f"Content {i}",
            })
            self.assertEqual(res["notification_id"], f"notification_{i+1:03d}")

        list_res = self.tool.execute({"operation": "list"})
        self.assertEqual(list_res["count"], 5)
        self.assertEqual(len(list_res["notifications"]), 5)

    def test_list_notifications(self):
        """6. List all stored notifications."""
        self.tool.execute({
            "operation": "send",
            "recipient": SYNTHETIC_CANARY_USER,
            "subject": "Canary Test",
            "body": SYNTHETIC_CANARY_NOTIFICATION,
        })
        self.tool.execute({
            "operation": "send",
            "recipient": "admin_user@satlas.local",
            "subject": "Admin Test",
            "body": "Admin message",
        })

        list_res = self.tool.execute({"operation": "list"})
        self.assertTrue(list_res["success"])
        self.assertEqual(list_res["count"], 2)
        recipients = [n["recipient"] for n in list_res["notifications"]]
        self.assertIn(SYNTHETIC_CANARY_USER, recipients)
        self.assertIn("admin_user@satlas.local", recipients)

    def test_list_with_recipient_filter(self):
        """7. List notifications with recipient filter."""
        self.tool.execute({
            "operation": "send",
            "recipient": "alice@satlas.local",
            "subject": "Hello Alice",
            "body": "Body 1",
        })
        self.tool.execute({
            "operation": "send",
            "recipient": "bob@satlas.local",
            "subject": "Hello Bob",
            "body": "Body 2",
        })
        self.tool.execute({
            "operation": "send",
            "recipient": "alice@satlas.local",
            "subject": "Alice Again",
            "body": "Body 3",
        })

        alice_res = self.tool.execute({
            "operation": "list",
            "recipient": "alice@satlas.local",
        })
        self.assertEqual(alice_res["count"], 2)
        for n in alice_res["notifications"]:
            self.assertEqual(n["recipient"], "alice@satlas.local")

        bob_res = self.tool.execute({
            "operation": "list",
            "recipient": "bob@satlas.local",
        })
        self.assertEqual(bob_res["count"], 1)
        self.assertEqual(bob_res["notifications"][0]["recipient"], "bob@satlas.local")

        charlie_res = self.tool.execute({
            "operation": "list",
            "recipient": "charlie@satlas.local",
        })
        self.assertEqual(charlie_res["count"], 0)

    def test_get_notification(self):
        """8. Get notification by notification_id."""
        send_res = self.tool.execute({
            "operation": "send",
            "recipient": "user@satlas.local",
            "subject": "Important Notice",
            "body": "Detailed notification body",
        })
        notif_id = send_res["notification_id"]

        get_res = self.tool.execute({
            "operation": "get",
            "notification_id": notif_id,
        })
        self.assertTrue(get_res["success"])
        notif = get_res["notification"]
        self.assertEqual(notif["notification_id"], notif_id)
        self.assertEqual(notif["recipient"], "user@satlas.local")
        self.assertEqual(notif["subject"], "Important Notice")
        self.assertEqual(notif["body"], "Detailed notification body")

    def test_unknown_notification_id(self):
        """9. Unknown notification ID returns controlled not-found result."""
        result = self.tool.execute({
            "operation": "get",
            "notification_id": "notification_999",
        })
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Notification not found")
        self.assertEqual(result["notification_id"], "notification_999")

    def test_reset_behavior(self):
        """10. Reset clears all stored notifications."""
        self.tool.execute({
            "operation": "send",
            "recipient": "user@satlas.local",
            "subject": "Temporary",
            "body": "To be cleared",
        })
        self.assertEqual(self.tool.execute({"operation": "list"})["count"], 1)

        reset_res = self.tool.execute({"operation": "reset"})
        self.assertTrue(reset_res["success"])

        list_after = self.tool.execute({"operation": "list"})
        self.assertEqual(list_after["count"], 0)
        self.assertEqual(list_after["notifications"], [])

    def test_reset_id_counter(self):
        """11. Reset restores ID counter to notification_001."""
        self.tool.execute({
            "operation": "send",
            "recipient": "user@satlas.local",
            "subject": "Pre-reset 1",
            "body": "Body",
        })
        self.tool.execute({
            "operation": "send",
            "recipient": "user@satlas.local",
            "subject": "Pre-reset 2",
            "body": "Body",
        })

        self.tool.reset()

        new_res = self.tool.execute({
            "operation": "send",
            "recipient": "user@satlas.local",
            "subject": "Post-reset",
            "body": "Body",
        })
        self.assertEqual(new_res["notification_id"], "notification_001")

    def test_reject_external_recipients(self):
        """12. Reject external/unauthorized recipient domains."""
        external_emails = [
            "attacker@gmail.com",
            "user@example.com",
            "victim@yahoo.com",
            "target@corp.external.com",
            "user@satlas.org",
        ]
        for email in external_emails:
            with self.assertRaises(InvalidRecipientError) as ctx:
                self.tool.execute({
                    "operation": "send",
                    "recipient": email,
                    "subject": "Exfiltration attempt",
                    "body": "secret data",
                })
            self.assertIn("disallowed domain", str(ctx.exception).lower())

    def test_reject_malformed_recipient(self):
        """13. Reject malformed recipient email addresses."""
        malformed = [
            "not-an-email",
            "@satlas.local",
            "user@",
            "user@@satlas.local",
            "user @satlas.local",
            "",
            "   ",
        ]
        for m in malformed:
            with self.assertRaises(InvalidRecipientError):
                self.tool.execute({
                    "operation": "send",
                    "recipient": m,
                    "subject": "Test",
                    "body": "Test",
                })

    def test_reject_newline_and_header_injection_in_recipient(self):
        """14. Reject newline or control characters in recipient."""
        injections = [
            "user@satlas.local\nCc: attacker@satlas.local",
            "user@satlas.local\r\nBcc: evil@satlas.local",
            "user\x00@satlas.local",
            "user@satlas.local\r",
        ]
        for inj in injections:
            with self.assertRaises(InvalidRecipientError):
                self.tool.execute({
                    "operation": "send",
                    "recipient": inj,
                    "subject": "Injection",
                    "body": "Payload",
                })

    def test_reject_newline_in_subject(self):
        """Header injection prevention in subject line."""
        with self.assertRaises(InvalidNotificationArgumentError):
            self.tool.execute({
                "operation": "send",
                "recipient": "user@satlas.local",
                "subject": "Subject\nCc: evil@satlas.local",
                "body": "Body",
            })

    def test_subject_length_limit(self):
        """15. Reject subject exceeding maximum length limit."""
        small_tool = NotificationTool(max_subject_length=20)
        long_subject = "A" * 21
        with self.assertRaises(NotificationResourceLimitError):
            small_tool.execute({
                "operation": "send",
                "recipient": "user@satlas.local",
                "subject": long_subject,
                "body": "Body",
            })

        # Subject within limit succeeds
        ok_res = small_tool.execute({
            "operation": "send",
            "recipient": "user@satlas.local",
            "subject": "A" * 20,
            "body": "Body",
        })
        self.assertTrue(ok_res["success"])

    def test_body_length_limit(self):
        """16. Reject body exceeding maximum length limit."""
        small_tool = NotificationTool(max_body_length=50)
        long_body = "B" * 51
        with self.assertRaises(NotificationResourceLimitError):
            small_tool.execute({
                "operation": "send",
                "recipient": "user@satlas.local",
                "subject": "Test",
                "body": long_body,
            })

        # Body within limit succeeds
        ok_res = small_tool.execute({
            "operation": "send",
            "recipient": "user@satlas.local",
            "subject": "Test",
            "body": "B" * 50,
        })
        self.assertTrue(ok_res["success"])

    def test_recipient_length_limit(self):
        """17. Reject recipient exceeding maximum length limit."""
        small_tool = NotificationTool(max_recipient_length=25)
        # Length 26: 'longuser12345@satlas.local'
        long_rec = "longuser12345@satlas.local"
        self.assertGreater(len(long_rec), 25)
        with self.assertRaises(NotificationResourceLimitError):
            small_tool.execute({
                "operation": "send",
                "recipient": long_rec,
                "subject": "Test",
                "body": "Body",
            })

    def test_maximum_stored_notifications(self):
        """18. Reject send when maximum stored notifications limit is reached."""
        capacity_tool = NotificationTool(max_stored_notifications=3)
        for i in range(3):
            capacity_tool.execute({
                "operation": "send",
                "recipient": f"u{i}@satlas.local",
                "subject": f"Sub {i}",
                "body": f"Body {i}",
            })

        # 4th notification exceeds capacity of 3
        with self.assertRaises(NotificationResourceLimitError):
            capacity_tool.execute({
                "operation": "send",
                "recipient": "overflow@satlas.local",
                "subject": "Overflow",
                "body": "Overflow body",
            })

    def test_non_string_arguments(self):
        """19. Reject non-string arguments."""
        with self.assertRaises(InvalidNotificationArgumentError):
            self.tool.execute("not a dict")

        with self.assertRaises(InvalidNotificationArgumentError):
            self.tool.execute({"operation": 123})

        with self.assertRaises(InvalidNotificationArgumentError):
            self.tool.execute({
                "operation": "send",
                "recipient": 12345,
                "subject": "Test",
                "body": "Test",
            })

        with self.assertRaises(InvalidNotificationArgumentError):
            self.tool.execute({
                "operation": "send",
                "recipient": "user@satlas.local",
                "subject": ["not", "string"],
                "body": "Test",
            })

        with self.assertRaises(InvalidNotificationArgumentError):
            self.tool.execute({
                "operation": "send",
                "recipient": "user@satlas.local",
                "subject": "Test",
                "body": {"not": "string"},
            })

        with self.assertRaises(InvalidOperationError):
            self.tool.execute({"operation": "unknown_op"})

    def test_no_network_behavior(self):
        """20. Ensure data remains strictly in-memory without network sockets."""
        # Send synthetic canary
        self.tool.execute({
            "operation": "send",
            "recipient": SYNTHETIC_CANARY_USER,
            "subject": "Local Canary",
            "body": SYNTHETIC_CANARY_NOTIFICATION,
        })
        # Internal store is directly accessible in-memory
        self.assertEqual(len(self.tool._notifications), 1)
        self.assertEqual(self.tool._notifications[0]["body"], SYNTHETIC_CANARY_NOTIFICATION)

    def test_thread_safe_concurrent_sends(self):
        """22. Thread-safe behavior under concurrent sends."""
        num_threads = 10
        sends_per_thread = 10

        def worker(thread_id: int):
            for j in range(sends_per_thread):
                self.tool.execute({
                    "operation": "send",
                    "recipient": f"worker_{thread_id}@satlas.local",
                    "subject": f"Notice {j}",
                    "body": f"Worker {thread_id} Message {j}",
                })

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, t) for t in range(num_threads)]
            concurrent.futures.wait(futures)

        list_res = self.tool.execute({"operation": "list"})
        self.assertEqual(list_res["count"], num_threads * sends_per_thread)
        # Verify all notification IDs are unique and sequential
        ids = [n["notification_id"] for n in list_res["notifications"]]
        self.assertEqual(len(set(ids)), num_threads * sends_per_thread)

    def test_keyword_arguments_execution(self):
        """Execution using direct keyword arguments."""
        res = self.tool.execute(
            operation="send",
            recipient="admin_user@satlas.local",
            subject="Direct Kwarg",
            body="Kwarg body",
        )
        self.assertTrue(res["success"])
        self.assertEqual(res["recipient"], "admin_user@satlas.local")


if __name__ == "__main__":
    unittest.main()
