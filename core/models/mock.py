from typing import List, Optional, Sequence
from core.models.base import ModelAdapter


class MockModel(ModelAdapter):
    """
    Deterministic mock model adapter for testing.

    Returns predefined responses regardless of the input prompt,
    enabling reproducible security and integration tests without network access.
    Supports a single fixed response or a sequence of responses for multi-step tests.
    """

    def __init__(
        self,
        response: str = "",
        responses: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Initialize the mock model with a fixed response or sequence of responses.

        Args:
            response: The deterministic response string to return on generate().
            responses: Optional sequence of response strings to return sequentially.
        """
        self.response = response
        self._responses: List[str] = list(responses) if responses is not None else []
        self._index = 0

    def set_response(self, response: str) -> None:
        """Update the fixed response and clear sequential responses."""
        self.response = response
        self._responses = []
        self._index = 0

    def set_responses(self, responses: Sequence[str]) -> None:
        """Set a new sequence of responses to return on subsequent generate() calls."""
        self._responses = list(responses)
        self._index = 0

    def generate(self, prompt: str) -> str:
        """
        Generate a response by returning predetermined response strings.

        Args:
            prompt: Input prompt provided to the model.

        Returns:
            The predefined response string.
        """
        if self._responses:
            if self._index < len(self._responses):
                res = self._responses[self._index]
                self._index += 1
                return res
            return self._responses[-1]
        return self.response
