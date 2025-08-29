import os
import json
from datetime import datetime

# Default per-million token prices in USD. Override via env vars if needed.
# Example env overrides:
#   OPENAI_PRICE_GPT_4O_INPUT=5.0
#   OPENAI_PRICE_GPT_4O_OUTPUT=15.0
PRICES_PER_MILLION = {
    "gpt-4o": {"input": 5.0, "output": 15.0},
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


def record_chat_completion_cost(resp, model: str, tag: str = "") -> dict:
    """
    Extract usage from a v1 chat completion response and append to Outputs/costs.log.
    Returns the computed dict with tokens and costs for convenience.
    """
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
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
