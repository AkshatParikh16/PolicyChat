from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """
    Abstract base class for all LLM providers.
    Forces every LLM to implement the same interface.
    """
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from a prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM service is running."""
        pass