import os
import json
from datetime import datetime

# Default per-million token prices in USD. Override via env vars if needed.
# Example env overrides:
#   OPENAI_PRICE_GPT_4O_INPUT=5.0
#   OPENAI_PRICE_GPT_4O_OUTPUT=15.0
PRICES_PER_MILLION = {
    # GPT-4o family
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "azure-gpt-4o": {"input": 2.5, "output": 10.0},
    
    # GPT-4 family
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4-turbo-preview": {"input": 10.0, "output": 30.0},
    
    # GPT-3.5 family
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    
    # O1 family (reasoning models)
    "o1-preview": {"input": 15.0, "output": 60.0},
    "o1-mini": {"input": 3.0, "output": 12.0},
    
    # Local/Ollama models (approximate cost as zero for local inference)
    "gpt-oss:20b": {"input": 0.0, "output": 0.0},
    "llama3.2:3b": {"input": 0.0, "output": 0.0},
}


def _env_price_key(model: str, kind: str) -> str:
    # kind: "INPUT" | "OUTPUT"
    return f"OPENAI_PRICE_{model.replace('-', '_').upper()}_{kind.upper()}"


def _get_price_per_million(model: str, kind: str) -> float:
    # Kind is 'input' or 'output'
    env_key = _env_price_key(model, kind)
    if env_key in os.environ:
        try:
            return float(os.environ[env_key])
        except ValueError:
            raise ValueError(f"Invalid model and kind: {env_key}")
    entry = PRICES_PER_MILLION.get(model, {})
    # Fallback: if unknown model, use gpt-4o as a conservative default
    if not entry:
        entry = PRICES_PER_MILLION["gpt-4o"]
    return float(entry.get(kind, 0.0))


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> dict:
    price_in = _get_price_per_million(model, "input")
    price_out = _get_price_per_million(model, "output")
    prompt_cost = (prompt_tokens or 0) / 1_000_000.0 * price_in
    completion_cost = (completion_tokens or 0) / 1_000_000.0 * price_out
    total_cost = prompt_cost + completion_cost
    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int((prompt_tokens or 0) + (completion_tokens or 0)),
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": total_cost,
    }


def record_chat_completion_cost(resp=None, model: str = None, tag: str = "", usage_dict: dict = None) -> dict:
    """
    Extract usage and append to Outputs/costs.log.
    
    Args:
        resp: OpenAI response object (legacy support) or None
        model: Model name (required)
        tag: Tag for logging
        usage_dict: Dictionary with 'input_tokens'/'output_tokens' or 'prompt_tokens'/'completion_tokens'
    
    Returns the computed dict with tokens and costs for convenience.
    """
    # Extract tokens from either usage_dict or response object
    if usage_dict:
        # Handle both naming conventions
        prompt_tokens = usage_dict.get('prompt_tokens', usage_dict.get('input_tokens', 0))
        completion_tokens = usage_dict.get('completion_tokens', usage_dict.get('output_tokens', 0))
    elif resp:
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    else:
        prompt_tokens = 0
        completion_tokens = 0
    
    info = estimate_cost(model, prompt_tokens, completion_tokens)

    os.makedirs("Outputs", exist_ok=True)
    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "model": model,
        "tag": tag,
        **info,
    }
    try:
        with open("Outputs/costs.log", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        # Non-fatal if logging fails
        pass
    return entry
