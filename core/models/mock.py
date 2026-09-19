from core.models.base import ModelAdapter


class MockModel(ModelAdapter):
    """
    Deterministic mock model adapter for testing.

    Returns a predefined response regardless of the input prompt,
    enabling reproducible security and integration tests without network access.
    """

    def __init__(self, response: str = "") -> None:
        """
        Initialize the mock model with a fixed response.

        Args:
            response: The deterministic response string to return on generate().
        """
        self.response = response

    def generate(self, prompt: str) -> str:
        """
        Generate a response by returning the predetermined response string.

        Args:
            prompt: Input prompt provided to the model.

        Returns:
            The predefined response string.
        """
        return self.response
