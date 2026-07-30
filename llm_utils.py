"""
Refactored LLM utility module for GeneAgent using BaseAgent infrastructure.

This module provides a simplified interface for LLM operations by directly
using BaseAgent's get_llm() function and LangChain's native features.

Key improvements:
- Uses LangChain's native tool binding exclusively (no OpenAI client dependency)
- Consistent BaseAgent integration throughout
- Simplified architecture with better maintainability
- Full support for all providers through BaseAgent
"""

import os
import json
import re
import uuid
import requests
from typing import Optional, Tuple
from dotenv import load_dotenv

from BaseAgent.llm import get_llm, extract_usage_metrics, SourceType, UsageMetrics
from BaseAgent.config import default_config
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

load_dotenv()


def _repair_json_string(json_str: str) -> str:
    """
    Repair common JSON formatting errors from Ollama models.

    Common issues:
    - Extra closing braces: {"key": "val"}}
    - Trailing commas: {"key": "val",}
    - Extra spaces/newlines
    """
    json_str = json_str.strip()

    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)

    while json_str.endswith('}}') and json_str.count('{') < json_str.count('}'):
        json_str = json_str[:-1]

    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    if open_braces > close_braces:
        json_str += '}' * (open_braces - close_braces)
    elif close_braces > open_braces:
        extra = close_braces - open_braces
        for _ in range(extra):
            if json_str.endswith('}'):
                json_str = json_str[:-1]

    return json_str


def _extract_and_repair_tool_call(error_message: str, available_functions: dict) -> Optional[dict]:
    """
    Extract tool call arguments from an Ollama tool-call parsing error and repair them.

    Ollama's client-side tool-call parser can fail on models with weaker native tool
    support (e.g. gpt-oss), raising errors like:
    "error parsing tool call: raw='{"gene...", err=...".
    This recovers the intended call so the caller can proceed instead of losing the turn.

    Args:
        error_message: The exception message raised by the bound LLM's invoke()
        available_functions: Dict of {function_name: openai_style_schema} used to
            infer which function was being called, since Ollama's error doesn't
            reliably include the function name in a parseable form.
    """
    match = re.search(r"raw='([^']+)'", error_message)
    if not match:
        return None

    repaired_json = _repair_json_string(match.group(1))

    try:
        parsed_args = json.loads(repaired_json)
    except json.JSONDecodeError:
        return None

    function_name = "unknown"
    arg_keys = set(parsed_args.keys())
    for func_name, func_schema in available_functions.items():
        param_keys = set(func_schema.get("parameters", {}).get("properties", {}).keys())
        if arg_keys.issubset(param_keys) or param_keys.issubset(arg_keys):
            function_name = func_name
            break

    return {
        "name": function_name,
        "args": parsed_args,
        "id": f"call_{uuid.uuid4().hex[:24]}",
    }


class SimpleLLMClient:
    """
    Simplified LLM client that uses BaseAgent's infrastructure directly.

    This client provides a clean interface for:
    - Chat completions using BaseAgent's get_llm()
    - Tool/function calling using LangChain's native .bind_tools()
    - Usage metrics extraction from BaseAgent
    - Consistent behavior across all providers (with fallback to OpenAI client for legacy support)
    """
    
    def __init__(
        self, 
        llm_model: str,
        temperature: float = 0,
    ):
        """
        Initialize the LLM client.
        
        Args:
            llm_model: The model name (e.g., "gpt-4o", "claude-3-5-sonnet-20241022")
        """
        self.llm_model = llm_model
        self.source, self.llm = get_llm(
            model=llm_model,
            temperature=temperature,
            stop_sequences=None,
            source=None,  # Auto-detect
            config=default_config,
        )
        # Fix model name for azure models
        if self.source == "AzureOpenAI":
            self.llm_model = self.llm_model.replace("azure-", "").replace("azure_", "")
        # BaseAgent hard-codes num_ctx=8192 for Ollama, which is too small for long
        # GeneAgent prompts (input alone can exceed 6k tokens). Mixtral 8x22b supports
        # up to 65536 — use 32768 as a safe default that fits in GPU memory.
        if self.source == "Ollama" and hasattr(self.llm, "num_ctx"):
            self.llm.num_ctx = 32768
    
    def chat(
        self, 
        messages: list[dict],
    ) -> Tuple[str, Optional[UsageMetrics]]:
        """
        Simple chat completion using BaseAgent's LLM.
        
        Args:
            messages: List of message dicts in OpenAI format
        
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
        
        # Invoke LLM
        response = self.llm.invoke(lc_messages)
        
        # Extract usage metrics
        usage_metrics = extract_usage_metrics(
            self.source,
            response,
            model=self.llm_model
        )
        
        # Get response content
        content = response.content if hasattr(response, 'content') else str(response)
        
        return content, usage_metrics
    
    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> Tuple[AIMessage, Optional[UsageMetrics]]:
        """
        Chat completion with tool calling using LangChain's native infrastructure.

        This method uses BaseAgent's LLM with .bind_tools() for consistent behavior
        across all providers (OpenAI, Anthropic, Ollama, etc.). For Ollama models with
        weak native tool-call support (e.g. gpt-oss), a malformed tool call is repaired
        from the raw error text rather than propagated as an exception.

        Args:
            messages: List of message dicts in LangChain format
            tools: List of OpenAI-style tool/function schemas

        Returns:
            Tuple of (response_message, usage_metrics)
        """
        # Convert messages to LangChain format
        lc_messages = self._convert_messages_to_langchain(messages)

        # Bind tools using LangChain's native method
        # LangChain accepts OpenAI-style schemas directly
        llm_with_tools = self.llm.bind_tools(tools)

        try:
            response = llm_with_tools.invoke(lc_messages)
        except Exception as e:
            error_msg = str(e)
            is_ollama_parse_error = self.source == "Ollama" and (
                "error parsing tool call" in error_msg or "invalid character" in error_msg
            )
            if not is_ollama_parse_error:
                raise

            available_functions = {t["name"]: t for t in tools}
            repaired_call = _extract_and_repair_tool_call(error_msg, available_functions)
            if repaired_call is None:
                raise

            response = AIMessage(content="", tool_calls=[repaired_call])

        # Extract usage metrics using BaseAgent's utility
        usage_metrics = extract_usage_metrics(
            self.source,
            response,
            model=self.llm_model
        )

        return response, usage_metrics

    def _convert_messages_to_langchain(self, messages: list[dict]) -> list:
        """
        Convert OpenAI-style message dicts to LangChain message objects.

        Args:
            messages: List of message dicts in OpenAI format

        Returns:
            List of LangChain message objects
        """
        lc_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                # Handle both regular messages and tool calls
                if "tool_calls" in msg:
                    lc_messages.append(AIMessage(
                        content=content or "",
                        tool_calls=msg["tool_calls"]
                    ))
                else:
                    lc_messages.append(AIMessage(content=content))
            elif role == "tool":
                # Tool response messages
                lc_messages.append(ToolMessage(
                    content=content,
                    tool_call_id=msg.get("tool_call_id", "")
                ))
            elif role == "function":
                # Legacy function response (convert to tool message)
                lc_messages.append(ToolMessage(
                    content=content,
                    tool_call_id=msg.get("name", "")  # Use function name as ID for legacy
                ))

        return lc_messages

    def cleanup_memory(self) -> bool:
        """Unload Ollama model from memory to free resources."""
        if self.source != "Ollama":
            return True  # Not Ollama, nothing to do

        try:
            # Get Ollama base URL from environment or use default
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            # Strip /v1 suffix if present (needed for native Ollama API)
            ollama_url = ollama_url.rstrip("/").replace("/v1", "")

            # Use the full model name as stored, which is what Ollama expects
            model_name = self.llm_model
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={"model": model_name, "prompt": "", "keep_alive": 0},
                timeout=10
            )

            # Check if the model was unloaded successfully
            if response.status_code == 200:
                print(f"✓ Unloaded Ollama model '{model_name}' from memory")
                return True
            else:
                print(f"⚠ Failed to unload model (status {response.status_code})")
                return False
        except requests.exceptions.ConnectionError:
            print(f"⚠ Could not connect to Ollama service at {ollama_url}")
            return False
        except Exception as e:
            print(f"⚠ Error unloading model: {e}")
            return False


# Global client cache
_llm_clients: dict[str, SimpleLLMClient] = {}


def get_llm_client(
    llm_model: str,
    temperature: float = 0,
    ) -> SimpleLLMClient:
    """
    Get or create a SimpleLLMClient instance for the given model.
    Uses caching to avoid recreating clients.

    Args:
        llm_model: The model name

    Returns:
        SimpleLLMClient instance
    """
    if llm_model not in _llm_clients:
        _llm_clients[llm_model] = SimpleLLMClient(llm_model, temperature)
    return _llm_clients[llm_model]


def cleanup_all_clients():
    """
    Clear the LLM client cache.
    This is useful when running multiple datasets in sequence.
    """
    _llm_clients.clear()

