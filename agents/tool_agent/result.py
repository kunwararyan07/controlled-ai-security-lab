from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from core.models.tool_call import ToolCall


@dataclass
class ToolExecutionStep:
    """
    Detailed structured record of an individual tool execution step in an agent interaction.

    Why multi-step evidence preservation is required for ASI02:
    In complex agent workflows and attack scenarios (such as confused deputy attacks or
    unintended tool chaining), an exploit may consist of multiple stages (e.g. Step 1:
    extract sensitive data using an authorized tool, Step 2: attempt exfiltration using a
    different tool). If Step 2 is denied by policy or fails, naive result aggregators that
    collapse execution to a single scalar status (`tool_executed = False`) mask the fact that
    Step 1 executed successfully and crossed an internal security boundary. Preserving
    complete per-step execution evidence ensures the research environment can accurately
    detect, trace, and evaluate multi-step tool misuse.
    """

    step_number: int = 1
    tool_name: str = ""
    tool_call: Optional[ToolCall] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    tool_arguments: Optional[Dict[str, Any]] = None
    authorization_allowed: bool = False
    authorization_denied: bool = False
    authorization_decision: Optional[str] = None
    authorization_reason: Optional[str] = None
    tool_executed: bool = False
    tool_result: Optional[Any] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if self.tool_arguments is not None and not self.arguments:
            self.arguments = self.tool_arguments
        elif self.tool_arguments is None and self.arguments:
            self.tool_arguments = self.arguments

        if self.authorization_decision is not None:
            if self.authorization_decision.upper() in ("ALLOW", "ALLOWED"):
                self.authorization_allowed = True
            elif self.authorization_decision.upper() in ("DENY", "DENIED"):
                self.authorization_denied = True
        else:
            if self.authorization_allowed:
                self.authorization_decision = "ALLOW"
            elif self.authorization_denied:
                self.authorization_decision = "DENY"
            else:
                self.authorization_decision = "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        """Convert step record to a dictionary representation."""
        return {
            "step_number": self.step_number,
            "tool_name": self.tool_name,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "arguments": self.arguments,
            "tool_arguments": self.tool_arguments,
            "authorization_allowed": self.authorization_allowed,
            "authorization_denied": self.authorization_denied,
            "authorization_decision": self.authorization_decision,
            "authorization_reason": self.authorization_reason,
            "tool_executed": self.tool_executed,
            "tool_result": self.tool_result,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class AgentResult:
    """
    Structured result of a ToolUsingAgent execution.

    Attributes:
        session_id: The session or request identifier for correlating events.
        final_response: The final response content (if normal response or after tool execution).
        tool_call: The structured tool call requested by the model, if any (reflects latest/terminal tool).
        authorization_allowed: Whether authorization explicitly permitted the tool call.
        authorization_denied: Whether authorization explicitly denied the tool call.
        authorization_reason: Explanation from policy if authorization was evaluated.
        tool_executed: Whether the latest tool was actually executed.
        tool_result: The output returned by the executed tool, if executed.
        error: Error message if an error occurred during execution.
        tool_calls: Complete ordered list of all tool calls requested during the interaction.
        tool_results: Complete ordered list of results from tools that were executed.
        steps: Detailed per-step execution records preserving complete multi-step history.
    """

    session_id: Optional[str] = None
    user_input: Optional[str] = None
    final_response: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    authorization_allowed: bool = False
    authorization_denied: bool = False
    authorization_reason: Optional[str] = None
    tool_executed: bool = False
    tool_result: Optional[Any] = None
    error: Optional[str] = None
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    model_durations: list = field(default_factory=list)
    tool_durations: list = field(default_factory=list)
    total_duration: Optional[float] = None
    steps: List[ToolExecutionStep] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.tool_call is not None and not self.tool_calls:
            self.tool_calls = [self.tool_call]
        elif self.tool_call is None and self.tool_calls:
            self.tool_call = self.tool_calls[-1]

        # Invariant: tool_result must only correspond to tool_call when tool_executed is True.
        # A failed or unexecuted tool call must never have a tool_result or inherit a previous result.
        if self.tool_executed:
            if self.tool_result is not None and not self.tool_results:
                self.tool_results = [self.tool_result]
            elif self.tool_result is None and self.tool_results:
                if len(self.tool_results) == len(self.tool_calls):
                    self.tool_result = self.tool_results[-1]
                else:
                    self.tool_result = None
        else:
            self.tool_result = None

        # Auto-populate steps if not explicitly provided but tool_calls exist
        if not self.steps and self.tool_calls:
            for idx, tc in enumerate(self.tool_calls, 1):
                t_exec = idx <= len(self.tool_results)
                t_res = self.tool_results[idx - 1] if t_exec else None
                is_last = (idx == len(self.tool_calls))
                self.steps.append(
                    ToolExecutionStep(
                        step_number=idx,
                        tool_name=tc.tool_name,
                        tool_call=tc,
                        arguments=tc.arguments,
                        authorization_allowed=self.authorization_allowed if is_last else True,
                        authorization_denied=self.authorization_denied if is_last else False,
                        authorization_reason=self.authorization_reason if is_last else None,
                        tool_executed=t_exec,
                        tool_result=t_res,
                        error=self.error if is_last and not t_exec else None,
                    )
                )

    @property
    def tool_requested(self) -> bool:
        """Whether a tool was requested by the model."""
        return self.tool_call is not None or len(self.tool_calls) > 0

    @property
    def executed_steps(self) -> List[ToolExecutionStep]:
        """List of steps that were actually executed."""
        return [s for s in self.steps if s.tool_executed]

    @property
    def denied_steps(self) -> List[ToolExecutionStep]:
        """List of steps that were denied by authorization."""
        return [s for s in self.steps if s.authorization_denied]

    @property
    def has_executed_steps(self) -> bool:
        """Whether at least one tool call in the interaction executed successfully."""
        return len(self.tool_results) > 0 or any(s.tool_executed for s in self.steps)

    @property
    def step_count(self) -> int:
        """Total number of execution steps in this interaction."""
        return max(len(self.steps), len(self.tool_calls))

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent result to a dictionary representation."""
        return {
            "session_id": self.session_id,
            "user_input": self.user_input,
            "final_response": self.final_response,
            "tool_call": self.tool_call.to_dict() if self.tool_call else None,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_requested": self.tool_requested,
            "authorization_allowed": self.authorization_allowed,
            "authorization_denied": self.authorization_denied,
            "authorization_reason": self.authorization_reason,
            "tool_executed": self.tool_executed,
            "tool_result": self.tool_result,
            "tool_results": self.tool_results,
            "error": self.error,
            "model_durations": self.model_durations,
            "tool_durations": self.tool_durations,
            "total_duration": self.total_duration,
            "steps": [s.to_dict() for s in self.steps],
        }

