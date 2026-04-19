"""Thin client for the OpenAI-compatible endpoint exposed by llama-server.

llama-server supports:
  • Standard OpenAI `/v1/chat/completions`.
  • `response_format: {"type": "json_schema", "json_schema": {...}}` for
    grammar-constrained decoding (enforced by llama.cpp's sampler — the model
    CANNOT emit invalid output for the given schema).
  • Tool calls via Hermes parsing (Qwen3's native format).

We use the AsyncOpenAI client for symmetry with future providers, but we pin
`base_url` to our local llama-server — no external API calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from .config import get_settings


@dataclass(frozen=True)
class LLMResponse:
    content: str                    # raw text (may be JSON string)
    parsed: dict[str, Any] | None   # parsed JSON if response_format was json
    tool_calls: list[dict[str, Any]]
    finish_reason: str | None
    elapsed_ms: float
    model: str


_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncOpenAI(
            base_url=str(settings.llm_base_url),
            api_key="llama-cpp-local",  # llama-server accepts anything; we auth via Traefik upstream
            timeout=settings.llm_request_timeout,
            max_retries=1,
        )
    return _client


def build_category_schema(allowed_slugs: list[str]) -> dict[str, Any]:
    """JSON-schema that the model's output is constrained to.

    Keeping it tight on purpose: if the schema is permissive the model uses
    that freedom, and downstream parsing suffers.
    """
    return {
        "name": "categorization",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["category_slug", "confidence", "reasoning"],
            "properties": {
                "category_slug": {
                    "type": "string",
                    "enum": allowed_slugs,
                    "description": "Exactly one slug from the provided enum. Do not invent new slugs.",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "reasoning": {
                    "type": "string",
                    "maxLength": 300,
                    "description": "Breve justificación en español, una o dos frases.",
                },
            },
        },
    }


async def classify(
    *,
    messages: list[dict[str, Any]],
    allowed_slugs: list[str],
    tools: list[dict[str, Any]] | None = None,
    thinking: bool = False,
) -> LLMResponse:
    """Call llama-server with schema-constrained output and optional tools.

    `thinking=True` wraps the last user message with Qwen3's control token so
    the server runs in thinking mode. `thinking=False` is the default for
    classification (faster, equally accurate on clear cases — see research notes).
    """
    import json
    import time

    settings = get_settings()
    client = get_client()

    payload: dict[str, Any] = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "response_format": {
            "type": "json_schema",
            "json_schema": build_category_schema(allowed_slugs),
        },
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if thinking:
        # Qwen3's recipe: append `/think` to the last user message. No-think
        # mode is the default inference path and uses less token budget.
        last = payload["messages"][-1]
        if last.get("role") == "user" and isinstance(last.get("content"), str):
            last["content"] = last["content"].rstrip() + "\n/think"

    t0 = time.perf_counter()
    resp = await client.chat.completions.create(**payload)  # type: ignore[arg-type]
    elapsed = (time.perf_counter() - t0) * 1000

    choice = resp.choices[0]
    content = choice.message.content or ""
    parsed: dict[str, Any] | None = None
    if content.strip().startswith("{"):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None

    tool_calls: list[dict[str, Any]] = []
    for tc in (choice.message.tool_calls or []):
        tool_calls.append(
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
        )

    return LLMResponse(
        content=content,
        parsed=parsed,
        tool_calls=tool_calls,
        finish_reason=choice.finish_reason,
        elapsed_ms=elapsed,
        model=resp.model,
    )
