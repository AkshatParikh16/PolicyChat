import httpx
from app.llm.base import BaseLLM
from app.config import config
from app.logger.logger import get_logger

logger = get_logger(__name__)

class OllamaLLM(BaseLLM):
    """LLM provider using local Ollama."""
    
    def __init__(self):
        self.base_url = config.OLLAMA_BASE_URL
        self.model = config.OLLAMA_MODEL
        self.timeout = 120  # 2 minutes for local inference
    
    def generate(self, prompt: str) -> str:
        """Send prompt to Ollama and get response."""
        try:
            logger.info(f"Calling Ollama: {self.model}")
            
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 512
                    }
                },
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "").strip()
            
            logger.info(f"Ollama response: {len(answer)} chars")
            return answer
            
        except httpx.TimeoutException:
            logger.error("Ollama request timed out")
            raise Exception("LLM timed out — try a shorter query")
            
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama")
            raise Exception("Ollama is not running. Start it with: ollama serve")
            
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = httpx.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False