class AuthorizationDeniedError(PermissionError):
    """
    Raised when a tool execution request is denied by an authorization policy.

    Attributes:
        tool_name: The name of the tool whose execution was denied.
        reason: The reason provided by the authorization policy for the denial.
    """

    def __init__(self, tool_name: str, reason: str) -> None:
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Authorization denied for tool '{tool_name}': {reason}")
