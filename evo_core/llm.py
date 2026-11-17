"""Unified OpenAI-compatible LLM client with retry, history repair, and token accounting."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments_raw: str
    arguments: dict[str, Any]
    parse_error: str | None = None


@dataclass
class LLMResponse:
    content: str
    reasoning_content: str
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    raw_assistant_message: dict[str, Any] = field(default_factory=dict)


def redact_secret(text: str, secret: str | None) -> str:
    if not secret or not text:
        return text
    return text.replace(secret, "[REDACTED_API_KEY]")


def approximate_tokens(messages: list[dict[str, Any]]) -> int:
    """Fast heuristic token estimate (~3.5 chars/token) for compaction triggers."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif content is not None:
            total_chars += len(json.dumps(content, ensure_ascii=False, default=str))
        if msg.get("tool_calls"):
            total_chars += len(json.dumps(msg["tool_calls"], ensure_ascii=False, default=str))
    return max(1, int(total_chars / 3.5))


def repair_messages(
    messages: list[dict[str, Any]],
    *,
    preserve_reasoning: bool = False,
) -> list[dict[str, Any]]:
    """Normalize tool_call_ids, close dangling tool calls, and drop orphan tool messages."""
    repaired: list[dict[str, Any]] = []
    pending: dict[str, str] = {}

    def close_pending() -> None:
        for tc_id, tc_name in pending.items():
            repaired.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": tc_name,
                    "content": (
                        "Tool execution was interrupted before completing. "
                        "Inspect current files/state before retrying."
                    ),
                }
            )
        pending.clear()

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role == "tool":
            tc_id = msg.get("tool_call_id")
            if isinstance(tc_id, str) and tc_id in pending:
                repaired.append(dict(msg))
                pending.pop(tc_id)
            continue

        if pending:
            close_pending()

        copied = dict(msg)
        if role == "assistant":
            raw_tcs = copied.get("tool_calls") or []
            clean_tcs: list[dict[str, Any]] = []
            for j, tc in enumerate(raw_tcs):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                name = fn.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                tc_id = tc.get("id")
                if not isinstance(tc_id, str) or not tc_id.strip():
                    tc_id = f"call_repair_{idx}_{j}"
                args = fn.get("arguments", "{}")
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                clean_tcs.append(
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    }
                )
                pending[tc_id] = name
            if clean_tcs:
                copied["tool_calls"] = clean_tcs
            else:
                copied.pop("tool_calls", None)
            if preserve_reasoning:
                copied.setdefault("reasoning_content", "")
            else:
                copied.pop("reasoning_content", None)
        repaired.append(copied)

    if pending:
        close_pending()
    return repaired


class ChatClient:
    """Thin, deterministic wrapper over OpenAI chat completions."""

    RETRYABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        default_model: str,
        thinking: str = "enabled",
        reasoning_effort: str = "high",
        temperature: float = 0.2,
        max_output_tokens: int = 12_000,
        timeout: float = 180.0,
        max_retries: int = 4,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=0,
        )

    def _is_deepseek_model(self, model: str) -> bool:
        return "deepseek" in model.lower() or "api.deepseek.com" in self.base_url.lower()

    def _build_kwargs(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None,
        thinking_override: str | None,
        response_format: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], bool]:
        thinking_mode = thinking_override or self.thinking
        is_ds = self._is_deepseek_model(model)
        preserve_reasoning = is_ds and thinking_mode == "enabled"

        clean_msgs = repair_messages(messages, preserve_reasoning=preserve_reasoning)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": clean_msgs,
            "max_tokens": max_tokens or self.max_output_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format

        if is_ds and thinking_mode in {"enabled", "disabled"}:
            kwargs["extra_body"] = {"thinking": {"type": thinking_mode}}
            if thinking_mode == "enabled":
                if self.reasoning_effort:
                    kwargs["reasoning_effort"] = self.reasoning_effort
            else:
                kwargs["temperature"] = self.temperature
        else:
            kwargs["temperature"] = self.temperature

        return kwargs, preserve_reasoning

    def _should_retry(self, exc: Exception) -> bool:
        if "WinError 10013" in str(exc) or "10013" in repr(getattr(exc, "__cause__", "")):
            return False
        if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, httpx.TransportError)):
            return True
        if isinstance(exc, APIStatusError):
            return exc.status_code in self.RETRYABLE_STATUSES
        return False

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        thinking_override: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        target_model = model or self.default_model
        kwargs, preserve_reasoning = self._build_kwargs(
            model=target_model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            thinking_override=thinking_override,
            response_format=response_format,
        )

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self._client.chat.completions.create(**kwargs)
                return self._parse_response(raw, preserve_reasoning=preserve_reasoning)
            except Exception as exc:
                last_exc = exc
                if attempt >= self.max_retries or not self._should_retry(exc):
                    msg = redact_secret(str(exc), self.api_key)
                    raise RuntimeError(f"LLM request failed ({type(exc).__name__}): {msg}") from exc
                wait = min(45.0, (2.0 ** attempt) + random.uniform(0.1, 1.0))
                time.sleep(wait)

        raise RuntimeError(f"LLM request failed: {redact_secret(str(last_exc), self.api_key)}")

    @staticmethod
    def _parse_response(raw: Any, *, preserve_reasoning: bool) -> LLMResponse:
        choice = raw.choices[0]
        msg = choice.message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""

        parsed_calls: list[ToolCallRequest] = []
        serialized_calls: list[dict[str, Any]] = []
        for idx, tc in enumerate(getattr(msg, "tool_calls", None) or []):
            tc_id = getattr(tc, "id", None) or f"call_{int(time.time() * 1000)}_{idx}"
            fn = tc.function
            name = getattr(fn, "name", "") or ""
            args_raw = getattr(fn, "arguments", "") or "{}"
            parse_err: str | None = None
            parsed_args: dict[str, Any] = {}
            try:
                loaded = json.loads(args_raw)
                if isinstance(loaded, dict):
                    parsed_args = loaded
                else:
                    parse_err = f"Expected JSON object arguments, got {type(loaded).__name__}"
            except json.JSONDecodeError as err:
                parse_err = f"Invalid JSON arguments: {err}"

            parsed_calls.append(
                ToolCallRequest(
                    id=tc_id,
                    name=name,
                    arguments_raw=args_raw,
                    arguments=parsed_args,
                    parse_error=parse_err,
                )
            )
            serialized_calls.append(
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args_raw},
                }
            )

        assistant_dict: dict[str, Any] = {"role": "assistant", "content": content}
        if serialized_calls:
            assistant_dict["tool_calls"] = serialized_calls
        if preserve_reasoning:
            assistant_dict["reasoning_content"] = reasoning

        usage = getattr(raw, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = int(getattr(details, "cached_tokens", 0) or 0) if details else 0

        return LLMResponse(
            content=content,
            reasoning_content=reasoning,
            tool_calls=parsed_calls,
            finish_reason=getattr(choice, "finish_reason", None) or "stop",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            raw_assistant_message=assistant_dict,
        )
