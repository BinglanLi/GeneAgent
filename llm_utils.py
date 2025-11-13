"""
Unified LLM utility module for GeneAgent.

This module provides a unified interface for LLM operations using BaseAgent's infrastructure,
while supporting both simple chat completions and function calling.
"""

import os
import json
from typing import Any, Optional
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from BaseAgent.llm import get_llm, extract_usage_metrics, SourceType
from BaseAgent.config import default_config
from costs import record_chat_completion_cost

load_dotenv()


class UnifiedLLMClient:
    """
    Unified LLM client that uses BaseAgent's infrastructure.
    Supports both simple chat completions and function calling.
    """
    
    def __init__(self, llm_model: str):
        """
        Initialize the unified LLM client.
        
        Args:
            llm_model: The model name (e.g., "gpt-4o", "gpt-oss:20b")
        """
        self.llm_model = llm_model
        self.source, self.llm = self._create_llm()
        self._openai_client = None  # Lazy initialization for function calling
        
    def _create_llm(self):
        """Create LLM instance using BaseAgent's infrastructure.
        
        Relies on BaseAgent's built-in model detection logic (_detect_source).
        Handles special cases for Ollama and Azure models.
        """
        source = None
        base_url = None
        api_key = None
        
        model_lower = self.llm_model.lower() if self.llm_model else ""
        
        # Special handling for Ollama models using OpenAI-compatible API
        # BaseAgent detects "gpt-oss" as Ollama source, but we want OpenAI-compatible endpoint
        if "gpt-oss" in model_lower:
            # Use Custom source with Ollama's OpenAI-compatible endpoint
            source = "Custom"
            base_url = "http://localhost:11434/v1"
            api_key = "ollama"
        
        # Let BaseAgent's get_llm handle all other detection via _detect_source()
        # It will auto-detect from model name if source is None
        source, llm = get_llm(
            model=self.llm_model,
            temperature=0,
            stop_sequences=None,
            source=source,  # None for auto-detection, or Custom for Ollama OpenAI-compatible
            base_url=base_url,
            api_key=api_key,
            config=default_config,
        )
        
        return source, llm
    
    def _get_openai_client(self):
        """
        Get OpenAI client for function calling.
        Only works for OpenAI-compatible APIs.
        Detects provider based on model name only.
        """
        if self._openai_client is None:
            from openai import OpenAI
            try:
                from openai import AzureOpenAI
            except Exception:
                AzureOpenAI = None
            
            # Determine source based on model name (matching BaseAgent's detection)
            model_lower = self.llm_model.lower() if self.llm_model else ""
            
            # Special handling for Ollama models using OpenAI-compatible API
            if "gpt-oss" in model_lower:
                self._openai_client = OpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",
                )
                return self._openai_client
            
            # Check for Azure models
            if model_lower.startswith("azure-") or model_lower.startswith("azure_"):
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("AZURE_API_BASE")
                azure_api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_API_KEY")
                azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("AZURE_API_VERSION")
                if azure_endpoint and azure_api_key and azure_api_version and AzureOpenAI is not None:
                    self._openai_client = AzureOpenAI(
                        azure_endpoint=azure_endpoint,
                        api_key=azure_api_key,
                        api_version=azure_api_version,
                    )
                    return self._openai_client
            
            # Default: Standard OpenAI API (for gpt-*, o1-*, etc.)
            self._openai_client = OpenAI()
        
        return self._openai_client
    
    def chat_completion(self, messages: list[dict], **kwargs) -> tuple[Any, Optional[dict]]:
        """
        Perform a simple chat completion using BaseAgent's LLM.
        
        Args:
            messages: List of message dicts in OpenAI format
            **kwargs: Additional arguments (currently unused, for compatibility)
        
        Returns:
            Tuple of (response_object, usage_metrics_dict)
            Response object mimics OpenAI's response format for compatibility
        """
        # Convert messages to LangChain format
        langchain_messages = self._convert_openai_to_langchain(messages)
        
        # Invoke LLM
        response = self.llm.invoke(langchain_messages)
        
        # Extract usage metrics
        usage_metrics = extract_usage_metrics(
            self.source, 
            response, 
            model=self.llm_model
        )
        
        # Get response content
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Create OpenAI-compatible response object
        mock_response = self._create_mock_response(content, usage_metrics)
        
        return mock_response, usage_metrics
    
    def chat_completion_with_functions(
        self, 
        messages: list[dict], 
        functions: list[dict],
        **kwargs
    ) -> tuple[Any, Optional[dict]]:
        """
        Perform chat completion with function calling.
        Uses OpenAI client directly for function calling support.
        For gpt-5 models, uses the newer tool calling format instead of legacy functions.
        
        Args:
            messages: List of message dicts in OpenAI format
            functions: List of function definitions in OpenAI format
            **kwargs: Additional arguments (e.g., temperature)
        
        Returns:
            Tuple of (response_object, usage_metrics_dict)
        """
        client = self._get_openai_client()
        
        # Determine temperature: gpt-5 requires temperature=1, use provided or default to 0
        temperature = kwargs.get("temperature", 0)
        if self.llm_model.startswith("gpt-5") and temperature == 0:
            # gpt-5 only supports temperature=1 (default)
            temperature = 1.0
        
        # gpt-5 uses tool calling format, not legacy functions format
        if self.llm_model.startswith("gpt-5"):
            if temperature == 0:
                temperature = 1.0
            
            # Convert functions to tools format for gpt-5
            tools = [{"type": "function", "function": func} for func in functions]
            
            # Filter out function role messages (gpt-5 doesn't support them)
            filtered_messages = []
            for msg in messages:
                if msg.get("role") != "function":
                    filtered_messages.append(msg)
            
            completion = client.chat.completions.create(
                model=self.llm_model,
                messages=filtered_messages,
                tools=tools,
                temperature=temperature,
            )
        else:
            # Legacy function calling for other models
            completion = client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                functions=functions,
                temperature=kwargs.get("temperature", 0),
            )
        
        # Extract usage metrics
        usage_metrics = None
        if hasattr(completion, "usage"):
            usage_metrics = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
            }
        
        return completion, usage_metrics
    
    def _convert_openai_to_langchain(self, messages: list[dict]):
        """Convert OpenAI message format to LangChain message format."""
        langchain_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            # Note: OpenAI's "function" role is not directly supported in LangChain
            # For function calling, we use OpenAI client directly
        
        return langchain_messages
    
    def _create_mock_response(self, content: str, usage_metrics: Optional[Any]):
        """Create an OpenAI-compatible response object."""
        class MockUsage:
            def __init__(self, metrics):
                self.prompt_tokens = metrics.input_tokens if metrics else 0
                self.completion_tokens = metrics.output_tokens if metrics else 0
                self.total_tokens = metrics.total_tokens if metrics else 0
        
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


# Global instance cache
_llm_clients: dict[str, UnifiedLLMClient] = {}


def get_llm_client(llm_model: str) -> UnifiedLLMClient:
    """
    Get or create a UnifiedLLMClient instance for the given model.
    Uses caching to avoid recreating clients.
    
    Args:
        llm_model: The model name
    
    Returns:
        UnifiedLLMClient instance
    """
    if llm_model not in _llm_clients:
        _llm_clients[llm_model] = UnifiedLLMClient(llm_model)
    return _llm_clients[llm_model]

