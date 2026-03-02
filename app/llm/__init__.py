from app.llm.ollama_llm import OllamaLLM
from app.config import config
from app.logger.logger import get_logger

logger = get_logger(__name__)


def get_llm():
    if config.LLM_PROVIDER == "ollama":
        logger.info("Using LLM provider: Ollama")
        return OllamaLLM()
    else:
        raise ValueError(f"Unknown LLM provider: {config.LLM_PROVIDER}")