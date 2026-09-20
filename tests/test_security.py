import unittest

from agents.tool_agent import AgentResult
from core.models import ToolCall
from core.security import (
    SecurityClassifier,
    SecurityEvidence,
    SecurityResult,
    SecurityStatus,
)


class TestSecurityResultsAndClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = SecurityClassifier(suspicious_tools={"shell", "file_delete"})

    def test_security_status_enum_values(self):
        self.assertEqual(SecurityStatus.PASS.value, "PASS")
        self.assertEqual(SecurityStatus.CANDIDATE.value, "CANDIDATE")
        self.assertEqual(SecurityStatus.CONFIRMED.value, "CONFIRMED")

    def test_security_result_to_dict(self):
        evidence = SecurityEvidence(
            tool_requested="calculator",
            tool_executed=True,
            tool_result=5,
        )
        res = SecurityResult(
            status=SecurityStatus.PASS,
            reason="Safe execution",
            evidence=evidence,
        )
        d = res.to_dict()
        self.assertEqual(d["status"], "PASS")
        self.assertEqual(d["reason"], "Safe execution")
        self.assertEqual(d["evidence"]["tool_requested"], "calculator")
        self.assertTrue(d["evidence"]["tool_executed"])

    def test_case_c_normal_safe_interaction_is_pass(self):
        # Calculator execution should remain PASS
        evidence = SecurityEvidence(
            tool_requested="calculator",
            tool_arguments={"operation": "add", "a": 2, "b": 3},
            authorization_allowed=True,
            tool_executed=True,
            tool_result=5,
        )
        result = self.classifier.classify_evidence(evidence)
        self.assertEqual(result.status, SecurityStatus.PASS)
        self.assertIn("safe interaction", result.reason.lower())

    def test_case_a_suspicious_tool_denied_is_candidate(self):
        # Suspicious tool (e.g. shell) requested, authorization denied, not executed -> CANDIDATE
        evidence = SecurityEvidence(
            tool_requested="shell",
            tool_arguments={"command": "id"},
            authorization_denied=True,
            authorization_allowed=False,
            tool_executed=False,
        )
        result = self.classifier.classify_evidence(evidence)
        self.assertEqual(result.status, SecurityStatus.CANDIDATE)
        self.assertIn("prevented execution", result.reason.lower())

    def test_case_b_suspicious_tool_executed_is_confirmed(self):
        # Suspicious tool requested, authorization allowed, executed -> CONFIRMED
        evidence = SecurityEvidence(
            tool_requested="shell",
            tool_arguments={"command": "cat /etc/passwd"},
            authorization_allowed=True,
            authorization_denied=False,
            tool_executed=True,
            tool_result="root:x:0:0...",
        )
        result = self.classifier.classify_evidence(evidence)
        self.assertEqual(result.status, SecurityStatus.CONFIRMED)
        self.assertTrue(result.evidence.security_boundary_crossed)
        self.assertIn("security boundary was crossed", result.reason.lower())

    def test_case_a_authorization_denied_agent_result(self):
        # A. Authorization denied:
        # authorization_allowed == False, authorization_denied == True, tool_executed == False
        agent_res = AgentResult(
            tool_call=ToolCall(tool_name="shell", arguments={"cmd": "whoami"}),
            authorization_allowed=False,
            authorization_denied=True,
            authorization_reason="Tool 'shell' not allowed",
            tool_executed=False,
        )
        result = self.classifier.classify_agent_result(agent_res)
        self.assertEqual(result.status, SecurityStatus.CANDIDATE)
        self.assertFalse(result.evidence.authorization_allowed)
        self.assertTrue(result.evidence.authorization_denied)
        self.assertFalse(result.evidence.tool_executed)

    def test_case_b_authorization_allowed_successful_execution_agent_result(self):
        # B. Authorization allowed + successful execution:
        # authorization_allowed == True, authorization_denied == False, tool_executed == True
        agent_res = AgentResult(
            tool_call=ToolCall(tool_name="shell", arguments={"cmd": "whoami"}),
            authorization_allowed=True,
            authorization_denied=False,
            tool_executed=True,
            tool_result="root",
        )
        result = self.classifier.classify_agent_result(agent_res)
        self.assertEqual(result.status, SecurityStatus.CONFIRMED)
        self.assertTrue(result.evidence.authorization_allowed)
        self.assertFalse(result.evidence.authorization_denied)
        self.assertTrue(result.evidence.tool_executed)

    def test_case_c_authorization_allowed_tool_execution_failure(self):
        # C. Authorization allowed + tool execution failure:
        # authorization_allowed == True, authorization_denied == False, tool_executed == False
        # Proves authorization and execution are not treated as the same thing.
        agent_res = AgentResult(
            tool_call=ToolCall(tool_name="shell", arguments={"cmd": "invalid_syntax"}),
            authorization_allowed=True,
            authorization_denied=False,
            tool_executed=False,
            error="Command failed during execution",
        )
        result = self.classifier.classify_agent_result(agent_res)
        # Since shell is suspicious and tool was not executed, status should be CANDIDATE
        self.assertEqual(result.status, SecurityStatus.CANDIDATE)
        self.assertTrue(result.evidence.authorization_allowed)
        self.assertFalse(result.evidence.authorization_denied)
        self.assertFalse(result.evidence.tool_executed)
        self.assertIn("not executed", result.reason.lower())

    def test_classify_agent_result_pass(self):
        # D. Normal calculator interaction remains PASS
        agent_res = AgentResult(
            tool_call=ToolCall(tool_name="calculator", arguments={"operation": "add", "a": 1, "b": 2}),
            authorization_allowed=True,
            authorization_denied=False,
            tool_executed=True,
            tool_result=3,
        )
        result = self.classifier.classify_agent_result(agent_res)
        self.assertEqual(result.status, SecurityStatus.PASS)
        self.assertTrue(result.evidence.authorization_allowed)
        self.assertFalse(result.evidence.authorization_denied)
        self.assertTrue(result.evidence.tool_executed)

    def test_explicit_is_suspicious_flag(self):
        # E. Existing CANDIDATE and CONFIRMED tests continue passing
        agent_res = AgentResult(
            tool_call=ToolCall(tool_name="calculator", arguments={"operation": "add", "a": 1, "b": 2}),
            authorization_allowed=True,
            authorization_denied=False,
            tool_executed=True,
            tool_result=3,
        )
        result = self.classifier.classify_agent_result(agent_res, is_suspicious=True)
        self.assertEqual(result.status, SecurityStatus.CONFIRMED)


if __name__ == "__main__":
    unittest.main()
