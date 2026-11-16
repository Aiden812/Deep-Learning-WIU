"""Optional LLM call for writing the answer. Configured in the repo's .env (see .env.example):

  LLM_PROVIDER=anthropic   LLM_API_KEY=...   LLM_MODEL=claude-haiku-5-5
  LLM_PROVIDER=openai      LLM_API_KEY=...   LLM_MODEL=...   LLM_BASE_URL=http://host:port/v1
                           (any OpenAI-compatible server, e.g. Daniel's llama-swap with Qwen)
  LLM_PROVIDER empty / none -> no LLM; answer.py falls back to quoting the best passages.

Only the de-identified question, the condition name and the retrieved passages are sent.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

TIMEOUT = 30


def provider() -> str:
    p = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    return p if p in {"anthropic", "openai"} and os.getenv("LLM_API_KEY") else "none"


def complete(system: str, user: str, max_tokens: int = 400) -> str:
    """Return the model's text. Raises on any failure so the caller can fall back."""
    p, key, model = provider(), os.getenv("LLM_API_KEY", ""), os.getenv("LLM_MODEL", "")
    if p == "anthropic":
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": model or "claude-haiku-5-5", "max_tokens": max_tokens, "temperature": 0.2,
                  "system": system, "messages": [{"role": "user", "content": user}]},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text").strip()
    if p == "openai":
        base = os.getenv("LLM_BASE_URL", "").rstrip("/")
        body = {"model": model, "max_tokens": max_tokens, "temperature": 0.2,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if base:  # a self-hosted server such as llama-swap + Qwen: answer directly, no thinking block
            body["chat_template_kwargs"] = {"enable_thinking": False}
        r = requests.post(
            f"{base or 'https://api.openai.com/v1'}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
            json=body,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError("no LLM configured")
