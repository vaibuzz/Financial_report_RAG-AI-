"""
Ollama local LLM provider.
"""

import json
from typing import List, Generator

import requests

from .base import BaseLLMProvider, Message, CompletionResponse


class OllamaProvider(BaseLLMProvider):
    """
    Provider for local Ollama instances.
    Ensures 100% local, zero-cost execution.
    """

    # Ollama implies 0 cost
    PRICING = {
        "default": {"prompt": 0.0, "completion": 0.0},
    }

    def __init__(self, api_key: str = "", model: str = "llama3", base_url: str = "http://localhost:11434", **kwargs):
        """
        Initialize Ollama provider.

        Args:
            api_key: Ignored for Ollama (kept for interface compatibility)
            model: Model name to use (e.g. "llama3", "mistral", "llama3.2")
            base_url: Ollama API base URL
            **kwargs: Provider-specific parameters
        """
        super().__init__(api_key, model, **kwargs)
        self.base_url = base_url.rstrip("/")

    def complete(
            self,
            messages: List[Message],
            temperature: float = 0.7,
            max_tokens: int = 1000,
            **kwargs
    ) -> CompletionResponse:
        """Generate synchronous completion using Ollama."""

        # Convert Message objects to Ollama format
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        # Merge extra params into options
        for k, v in kwargs.items():
            payload["options"][k] = v

        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            content = result.get("message", {}).get("content", "")
            
            # Ollama 0.1.29+ returns prompt_eval_count and eval_count
            tokens_prompt = result.get("prompt_eval_count", self.count_tokens("".join([msg.content for msg in messages])))
            tokens_completion = result.get("eval_count", self.count_tokens(content))
            tokens_total = tokens_prompt + tokens_completion
            
            return CompletionResponse(
                content=content,
                model=self.model,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                tokens_total=tokens_total,
                cost_usd=0.0,
                finish_reason="stop" if result.get("done") else "length"
            )
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama connection error: {str(e)}. Ensure Ollama is running at {self.base_url}")

    def stream(
            self,
            messages: List[Message],
            temperature: float = 0.7,
            max_tokens: int = 1000,
            **kwargs
    ) -> Generator[str, None, None]:
        """
        Generate streaming completion using Ollama.
        """

        # Convert Message objects
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        # Merge extra params into options
        for k, v in kwargs.items():
            payload["options"][k] = v

        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, stream=True, timeout=120)
            response.raise_for_status()

            # Accumulate response
            full_content = ""
            tokens_prompt = 0
            tokens_completion = 0

            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    
                    if "message" in data and "content" in data["message"]:
                        content = data["message"]["content"]
                        full_content += content
                        yield content
                    
                    if data.get("done", False):
                        # Capture token counts on final block
                        tokens_prompt = data.get("prompt_eval_count", 0)
                        tokens_completion = data.get("eval_count", 0)

            # Fallback for older versions
            if tokens_prompt == 0:
                tokens_prompt = self.count_tokens("".join([msg.content for msg in messages]))
            if tokens_completion == 0:
                tokens_completion = self.count_tokens(full_content)
                
            tokens_total = tokens_prompt + tokens_completion

            # Store the completion response
            self._last_completion_response = CompletionResponse(
                content=full_content,
                model=self.model,
                tokens_prompt=tokens_prompt,
                tokens_completion=tokens_completion,
                tokens_total=tokens_total,
                cost_usd=0.0,
                finish_reason="stop"
            )

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama streaming connection error: {str(e)}. Ensure Ollama is running at {self.base_url}")

    def count_tokens(self, text: str) -> int:
        """
        Approximate token count for local tracking.
        Ollama uses different tokenizers per model, so this is a rough estimate 
        (roughly 4 characters per token). Real token counts are returned in the response API.
        """
        return max(1, len(text) // 4)

    def calculate_cost(self, tokens_prompt: int, tokens_completion: int) -> float:
        """Cost is always zero for local execution."""
        return 0.0
