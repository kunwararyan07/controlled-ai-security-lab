import copy
import threading
from typing import Any, Dict, List, Optional

from core.interfaces.tool import Tool


class NotificationToolError(Exception):
    """Base exception for notification tool errors."""
    pass


class InvalidNotificationArgumentError(NotificationToolError, ValueError, TypeError):
    """Raised when arguments are missing, invalid, or improperly typed."""
    pass


class InvalidRecipientError(NotificationToolError, ValueError):
    """Raised when recipient email format or domain is invalid or disallowed."""
    pass


class NotificationNotFoundError(NotificationToolError, KeyError, ValueError):
    """Raised or referenced when a requested notification ID is not found."""
    pass


class NotificationResourceLimitError(NotificationToolError, ValueError):
    """Raised when notification size or store capacity limits are exceeded."""
    pass


class InvalidOperationError(NotificationToolError, ValueError):
    """Raised when an unsupported operation is requested."""
    pass


# Synthetic constants and fixtures
DEFAULT_ALLOWED_DOMAINS = ["satlas.local"]
SYNTHETIC_CANARY_NOTIFICATION = "SATLAS_CANARY_NOTIFICATION_001"
SYNTHETIC_CANARY_USER = "canary_user@satlas.local"
SYNTHETIC_REGULAR_USER = "regular_user@satlas.local"
SYNTHETIC_ADMIN_USER = "admin_user@satlas.local"


class NotificationTool(Tool):
    """
    Controlled, safe mock notification tool for security testing.

    Operates purely in-memory with synthetic recipients and data.
    Does NOT send real emails, connect to SMTP, or make network requests.
    Strictly restricts recipients to allowed synthetic domains (e.g. '.satlas.local').
    """

    def __init__(
        self,
        max_recipient_length: int = 256,
        max_subject_length: int = 512,
        max_body_length: int = 16384,
        max_stored_notifications: int = 1000,
        allowed_domains: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize NotificationTool with configurable resource limits.

        Args:
            max_recipient_length: Maximum allowed character length for recipient email.
            max_subject_length: Maximum allowed character length for notification subject.
            max_body_length: Maximum allowed character length for notification body.
            max_stored_notifications: Maximum number of notifications held in memory.
            allowed_domains: List of allowed synthetic domain suffixes (default: ['satlas.local']).
        """
        self.max_recipient_length = max_recipient_length
        self.max_subject_length = max_subject_length
        self.max_body_length = max_body_length
        self.max_stored_notifications = max_stored_notifications
        self.allowed_domains = (
            [d.lower().strip() for d in allowed_domains]
            if allowed_domains is not None
            else list(DEFAULT_ALLOWED_DOMAINS)
        )

        self._lock = threading.Lock()
        self._notifications: List[Dict[str, Any]] = []
        self._counter: int = 1

    @property
    def name(self) -> str:
        return "notification_tool"

    @property
    def description(self) -> str:
        return (
            "Performs controlled, synthetic notification operations (send, list, get, reset) "
            "strictly within an isolated in-memory mock store. Does NOT send real messages."
        )

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["send", "list", "get", "reset"],
                    "description": "The notification operation to perform ('send', 'list', 'get', 'reset').",
                },
                "recipient": {
                    "type": "string",
                    "description": "Synthetic recipient email address (required for 'send', optional for 'list'). Must belong to allowed synthetic domains (e.g. satlas.local).",
                },
                "subject": {
                    "type": "string",
                    "description": "Notification subject line (required for 'send').",
                },
                "body": {
                    "type": "string",
                    "description": "Notification message body (required for 'send').",
                },
                "notification_id": {
                    "type": "string",
                    "description": "Unique notification identifier (required for 'get').",
                },
            },
            "required": ["operation"],
        }

    def _validate_recipient(self, recipient: Any) -> str:
        """
        Validate recipient email format, domain, and character set.

        Raises:
            InvalidNotificationArgumentError: If recipient is not a string.
            NotificationResourceLimitError: If recipient length exceeds limit.
            InvalidRecipientError: If recipient format is invalid or domain is external.
        """
        if not isinstance(recipient, str):
            raise InvalidNotificationArgumentError(
                f"Recipient must be a string, got {type(recipient).__name__}."
            )

        if not recipient or not recipient.strip():
            raise InvalidRecipientError("Recipient cannot be empty.")

        # Reject control characters and newlines in the raw recipient string to prevent header injection
        for ch in recipient:
            if ord(ch) < 32 or ord(ch) == 127:
                raise InvalidRecipientError(
                    f"Recipient contains invalid control or newline character (ASCII {ord(ch)})."
                )

        # Disallow whitespace anywhere in the recipient email address
        if any(ch.isspace() for ch in recipient):
            raise InvalidRecipientError(f"Recipient '{recipient}' cannot contain whitespace.")

        if len(recipient) > self.max_recipient_length:
            raise NotificationResourceLimitError(
                f"Recipient length ({len(recipient)}) exceeds maximum allowed limit of {self.max_recipient_length}."
            )

        if "@" not in recipient:
            raise InvalidRecipientError(f"Recipient '{recipient}' is missing '@' symbol.")

        parts = recipient.split("@")
        if len(parts) != 2:
            raise InvalidRecipientError(f"Recipient '{recipient}' has invalid email format.")

        local_part, domain = parts
        domain = domain.lower()

        if not local_part:
            raise InvalidRecipientError(f"Recipient '{recipient}' has empty local part.")

        if not domain:
            raise InvalidRecipientError(f"Recipient '{recipient}' has empty domain.")

        domain_allowed = any(
            domain == allowed or domain.endswith("." + allowed)
            for allowed in self.allowed_domains
        )
        if not domain_allowed:
            raise InvalidRecipientError(
                f"External or disallowed domain '{domain}' rejected. "
                f"Notifications may only be sent to synthetic domains: {self.allowed_domains}."
            )

        return f"{local_part}@{domain}"

    def _validate_subject(self, subject: Any) -> str:
        """Validate subject string, length, and reject newlines."""
        if not isinstance(subject, str):
            raise InvalidNotificationArgumentError(
                f"Subject must be a string, got {type(subject).__name__}."
            )

        if len(subject) > self.max_subject_length:
            raise NotificationResourceLimitError(
                f"Subject length ({len(subject)}) exceeds maximum allowed limit of {self.max_subject_length}."
            )

        for ch in subject:
            if ch in ("\r", "\n", "\0"):
                raise InvalidNotificationArgumentError("Subject must not contain newline or control characters.")

        return subject

    def _validate_body(self, body: Any) -> str:
        """Validate body string and length."""
        if not isinstance(body, str):
            raise InvalidNotificationArgumentError(
                f"Body must be a string, got {type(body).__name__}."
            )

        if len(body) > self.max_body_length:
            raise NotificationResourceLimitError(
                f"Body length ({len(body)}) exceeds maximum allowed limit of {self.max_body_length}."
            )

        return body

    def execute(self, args: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
        """
        Execute the requested notification operation safely.

        Args:
            args: Dictionary of arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Structured dictionary describing operation result.
        """
        if args is not None:
            if not isinstance(args, dict):
                raise InvalidNotificationArgumentError("Arguments must be provided as a dictionary.")
            params = dict(args)
            params.update(kwargs)
        else:
            params = kwargs

        if not params:
            raise InvalidNotificationArgumentError("Missing arguments for notification_tool.")

        if "operation" not in params:
            raise InvalidNotificationArgumentError("Missing required argument: 'operation'.")

        operation = params["operation"]
        if not isinstance(operation, str):
            raise InvalidNotificationArgumentError("Argument 'operation' must be a string.")

        op = operation.strip().lower()

        if op == "send":
            return self._handle_send(params)
        elif op == "list":
            return self._handle_list(params)
        elif op == "get":
            return self._handle_get(params)
        elif op == "reset":
            return self._handle_reset()
        else:
            raise InvalidOperationError(
                f"Unsupported operation: '{operation}'. Supported operations: 'send', 'list', 'get', 'reset'."
            )

    def _handle_send(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a synthetic notification to in-memory store."""
        if "recipient" not in params:
            raise InvalidNotificationArgumentError("Missing required argument: 'recipient'.")
        if "subject" not in params:
            raise InvalidNotificationArgumentError("Missing required argument: 'subject'.")
        if "body" not in params:
            raise InvalidNotificationArgumentError("Missing required argument: 'body'.")

        recipient = self._validate_recipient(params["recipient"])
        subject = self._validate_subject(params["subject"])
        body = self._validate_body(params["body"])

        with self._lock:
            if len(self._notifications) >= self.max_stored_notifications:
                raise NotificationResourceLimitError(
                    f"Maximum stored notifications limit ({self.max_stored_notifications}) reached."
                )

            notification_id = f"notification_{self._counter:03d}"
            self._counter += 1

            notif = {
                "notification_id": notification_id,
                "recipient": recipient,
                "subject": subject,
                "body": body,
                "status": "queued",
            }
            self._notifications.append(notif)

            return {
                "success": True,
                "notification_id": notification_id,
                "recipient": recipient,
                "subject": subject,
                "status": "queued",
            }

    def _handle_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List stored synthetic notifications with optional recipient filter."""
        recipient_filter = params.get("recipient")
        filter_val: Optional[str] = None
        if recipient_filter is not None:
            if not isinstance(recipient_filter, str):
                raise InvalidNotificationArgumentError("Recipient filter must be a string.")
            filter_val = recipient_filter.strip().lower()

        with self._lock:
            if filter_val is not None:
                results = [
                    copy.deepcopy(n)
                    for n in self._notifications
                    if n["recipient"].lower() == filter_val
                ]
            else:
                results = [copy.deepcopy(n) for n in self._notifications]

            return {
                "success": True,
                "notifications": results,
                "count": len(results),
            }

    def _handle_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve a stored notification by notification_id."""
        if "notification_id" not in params:
            raise InvalidNotificationArgumentError("Missing required argument: 'notification_id'.")

        notification_id = params["notification_id"]
        if not isinstance(notification_id, str):
            raise InvalidNotificationArgumentError("Argument 'notification_id' must be a string.")

        target_id = notification_id.strip()
        if not target_id:
            raise InvalidNotificationArgumentError("Argument 'notification_id' cannot be empty.")

        with self._lock:
            notif = next(
                (n for n in self._notifications if n["notification_id"] == target_id),
                None,
            )
            if notif is None:
                return {
                    "success": False,
                    "error": "Notification not found",
                    "notification_id": target_id,
                }

            return {
                "success": True,
                "notification": copy.deepcopy(notif),
            }

    def _handle_reset(self) -> Dict[str, Any]:
        """Reset the notification store via execute()."""
        self.reset()
        return {
            "success": True,
            "message": "Notification store reset successfully.",
        }

    def reset(self) -> None:
        """
        Reset notification store to initial clean state.
        Clears all stored notifications and resets ID counter to 1.
        """
        with self._lock:
            self._notifications.clear()
            self._counter = 1
