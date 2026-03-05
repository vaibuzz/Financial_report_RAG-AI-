"""
Test script for the new OllamaProvider locally.
"""

import sys
import os

# Add the project root to the python path to import the rag module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag.providers.ollama import OllamaProvider
from rag.providers.base import Message

def test_ollama():
    print("Testing OllamaProvider initialized to localhost:11434 with llama3")
    
    try:
        provider = OllamaProvider(model="llama3", base_url="http://localhost:11434")
    except Exception as e:
        print(f"Error initializing provider: {e}")
        return

    messages = [
        Message(role="system", content="Sei un assistente AI brillante e divertente."),
        Message(role="user", content="Raccontami una barzelletta sulla finanza in una sola frase.")
    ]

    print("\n--- Testing Synchronous Completion ---")
    try:
        response = provider.complete(messages=messages, temperature=0.7, max_tokens=100)
        print(f"Content: {response.content}")
        print(f"Tokens Prompt: {response.tokens_prompt}")
        print(f"Tokens Completion: {response.tokens_completion}")
        print(f"Cost USD: ${response.cost_usd}")
        print(f"Finish Reason: {response.finish_reason}")
        
        assert response.cost_usd == 0.0, "Cost should be 0.0"
        
    except Exception as e:
        print(f"Sync Completion Error: {e}")

    print("\n--- Testing Streaming Completion ---")
    try:
        stream = provider.stream(messages=messages, temperature=0.7, max_tokens=100)
        
        print("Stream output: ", end="", flush=True)
        for chunk in stream:
            print(chunk, end="", flush=True)
            
        print()
        
        # Access the final response
        last_response = provider._last_completion_response
        print(f"\nStream Tokens Prompt: {last_response.tokens_prompt}")
        print(f"Stream Tokens Completion: {last_response.tokens_completion}")
        print(f"Stream Cost USD: ${last_response.cost_usd}")
        print(f"Stream Finish Reason: {last_response.finish_reason}")
        
        assert last_response.cost_usd == 0.0, "Cost should be 0.0"

    except Exception as e:
        print(f"\nStreaming Completion Error: {e}")


if __name__ == "__main__":
    test_ollama()
