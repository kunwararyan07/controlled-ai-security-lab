from abc import ABC, abstractmethod


class ModelAdapter(ABC):
    """
    Base interface for all LLM providers used by the security lab.
    """

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: Input prompt provided to the model.

        Returns:
            The model's raw response.
        """
        raise NotImplementedError