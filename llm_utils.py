"""
Refactored LLM utility module for GeneAgent using BaseAgent infrastructure.

This module provides a simplified interface for LLM operations by directly
using BaseAgent's get_llm() function without unnecessary wrappers.
"""

import json
from typing import Any, Optional, Tuple
from dotenv import load_dotenv

from openai import OpenAI, AzureOpenAI

from BaseAgent.llm import get_llm, extract_usage_metrics, SourceType, UsageMetrics
from BaseAgent.config import default_config
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()


class SimpleLLMClient:
    """
    Simplified LLM client that uses BaseAgent's infrastructure directly.
    
    This replaces the UnifiedLLMClient with a cleaner, simpler design that:
    - Uses BaseAgent's get_llm() for LangChain-based LLMs
    - Uses OpenAI client directly for function calling (when needed)
    - Minimal wrapper, maximum clarity
    """
    
    def __init__(self, llm_model: str):
        """
        Initialize the LLM client.
        
        Args:
            llm_model: The model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        """
        self.llm_model = llm_model
        self.source, self.llm = get_llm(
            model=llm_model,
            temperature=0,
            stop_sequences=None,
            source=None,  # Auto-detect
            config=default_config,
        )
        # Fix model name for azure models
        if self.source == "AzureOpenAI":
            self.llm_model = self.llm_model.replace("azure-", "").replace("azure_", "")
    
    def chat(
        self, 
        messages: list[dict],
        temperature: float = 0,
    ) -> Tuple[str, Optional[UsageMetrics]]:
        """
        Simple chat completion using BaseAgent's LLM.
        
        Args:
            messages: List of message dicts in OpenAI format
            temperature: Temperature for generation
        
        Returns:
            Tuple of (response_content, usage_metrics)
        """
        # Convert to LangChain message format
        lc_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        
        # If temperature is different from default, create a new LLM instance
        if temperature != 0:
            _, llm = get_llm(
                model=self.llm_model,
                temperature=temperature,
                stop_sequences=None,
                source=self.source,
                config=default_config,
            )
        else:
            llm = self.llm
        
        # Invoke LLM
        response = llm.invoke(lc_messages)
        
        # Extract usage metrics
        usage_metrics = extract_usage_metrics(
            self.source,
            response,
            model=self.llm_model
        )
        
        # Get response content
        content = response.content if hasattr(response, 'content') else str(response)
        
        return content, usage_metrics
    
    def chat_with_functions(
        self,
        messages: list[dict],
        functions: list[dict],
        temperature: float = 0,
    ) -> Tuple[Any, Optional[dict]]:
        """
        Chat completion with function calling using OpenAI client.
        
        Args:
            messages: List of message dicts in OpenAI format
            functions: List of function definitions
            temperature: Temperature for generation
        
        Returns:
            Tuple of (completion_object, usage_dict)
        """
        client = self._get_openai_client()
        
        # Handle gpt-5 models (use tool calling format)
        if self.llm_model.startswith("gpt-5"):
            temperature = 1.0 if temperature == 0 else temperature
            tools = [{"type": "function", "function": func} for func in functions]
            
            # Filter out function role messages
            filtered_messages = [
                msg for msg in messages if msg.get("role") != "function"
            ]
            
            completion = client.chat.completions.create(
                model=self.llm_model,
                messages=filtered_messages,
                tools=tools,
                temperature=temperature,
            )
        else:
            # Legacy function calling
            completion = client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                functions=functions,
                temperature=temperature,
            )
        
        # Extract usage
        usage_dict = None
        if hasattr(completion, "usage"):
            usage_dict = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }
        
        return completion, usage_dict
    
    def _get_openai_client(self):
        """Get OpenAI client for function calling."""
        import os
        
        model_lower = self.llm_model.lower()
        
        # Handle Ollama models - detect by common patterns
        # Patterns: "gpt-oss:model", "llama*", "model:tag" format
        is_ollama = (
            "gpt-oss" in model_lower or
            "llama" in model_lower or
            "mistral" in model_lower or
            "qwen" in model_lower or
            "phi" in model_lower or
            "gemma" in model_lower or
            self.source == "Ollama" or  # Trust the source detection
            (":" in model_lower and not model_lower.startswith("azure"))  # Common Ollama naming: model:tag
        )
        
        if is_ollama:
            return OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            )
        
        # Handle Azure OpenAI
        if model_lower.startswith("azure-") or model_lower.startswith("azure_"):
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_API_BASE")
            azure_api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY")
            azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("AZURE_API_VERSION")
            
            if azure_endpoint and azure_api_key and azure_api_version:
                return AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=azure_api_key,
                    api_version=azure_api_version,
                )
        
        # Default: Standard OpenAI
        return OpenAI()


# Global client cache
_llm_clients: dict[str, SimpleLLMClient] = {}


def get_llm_client(llm_model: str) -> SimpleLLMClient:
    """
    Get or create a SimpleLLMClient instance for the given model.
    Uses caching to avoid recreating clients.
    
    Args:
        llm_model: The model name
    
    Returns:
        SimpleLLMClient instance
    """
    if llm_model not in _llm_clients:
        _llm_clients[llm_model] = SimpleLLMClient(llm_model)
    return _llm_clients[llm_model]


def create_mock_openai_response(content: str, usage_metrics: Optional[UsageMetrics]) -> Any:
    """
    Create an OpenAI-compatible response object for backward compatibility.
    
    Args:
        content: Response content
        usage_metrics: Usage metrics from BaseAgent
    
    Returns:
        Mock response object with OpenAI-like structure
    """
    class MockUsage:
        def __init__(self, metrics):
            self.prompt_tokens = metrics.input_tokens if metrics and metrics.input_tokens else 0
            self.completion_tokens = metrics.output_tokens if metrics and metrics.output_tokens else 0
            self.total_tokens = metrics.total_tokens if metrics and metrics.total_tokens else 0
    
    class MockMessage:
        def __init__(self, content):
            self.content = content
            self.role = "assistant"
    
    class MockChoice:
        def __init__(self, content):
            self.message = MockMessage(content)
    
    class MockResponse:
        def __init__(self, content, usage_metrics):
            self.choices = [MockChoice(content)]
            self.usage = MockUsage(usage_metrics)
    
    return MockResponse(content, usage_metrics)
