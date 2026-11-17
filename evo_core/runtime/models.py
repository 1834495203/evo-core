"""Shared model telemetry. Provider SDK adapters may instead emit equivalent events."""
from dataclasses import asdict, replace
from pathlib import Path
import json
import time
import math
from threading import Lock
from ..llm import redact_secret
from .host import call_scope, CURRENT_CALL


class ObservedModels:
    def __init__(self, client, host, run_dir):
        self.__client, self.__host = client, host
        self.__root = Path(run_dir) / "model_calls"
        self.__sequence = 0
        self.counts = {}
        self.usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
        self.__lock = Lock()
        self.trace = None

    def __getattr__(self, name):
        return getattr(self.__client, name)

    def scope(self, actor):
        return call_scope(replace(CURRENT_CALL.get() or self.__host.identity, actor_id=actor))

    def counts_snapshot(self):
        with self.__lock: return dict(self.counts)

    def seed_counts(self, counts, usage=None):
        with self.__lock:
            self.counts = dict(counts)
            self.__sequence = sum(counts.values())
            if usage: self.usage.update(usage)

    def usage_snapshot(self):
        with self.__lock: return dict(self.usage)

    def report_sdk_call(self, *, input_tokens=0, output_tokens=0, cached_tokens=0, duration_ms=0, component="sdk", request_ref=None, response_ref=None):
        values = {"input_tokens": input_tokens, "output_tokens": output_tokens, "cached_tokens": cached_tokens}
        if any(type(v) is not int or v < 0 for v in values.values()):
            raise ValueError("SDK usage must contain nonnegative integer token counts")
        if type(duration_ms) not in {int, float} or not math.isfinite(duration_ms) or duration_ms < 0:
            raise ValueError("SDK duration must be finite and nonnegative")
        from .contracts import Result
        Result(artifacts=tuple(ref for ref in (request_ref, response_ref) if ref is not None)).validate()
        identity = self.__host.identify()
        with self.__lock:
            self.__sequence += 1
            sequence = self.__sequence
            self.counts[identity.actor_id] = self.counts.get(identity.actor_id, 0) + 1
            for k, v in values.items(): self.usage[k] += v
        self.__host.emit("llm_completed", identity=asdict(identity), sequence=sequence, source="sdk", component=component,
                         duration_ms=duration_ms, request_ref=asdict(request_ref) if request_ref else None,
                         response_ref=asdict(response_ref) if response_ref else None, **values)

    def complete(self, messages, **kwargs):
        identity = self.__host.identify()
        with self.__lock:
            self.__sequence += 1
            sequence = self.__sequence
            self.counts[identity.actor_id] = self.counts.get(identity.actor_id, 0) + 1
        with call_scope(identity):
            request = redact_secret(json.dumps(messages, ensure_ascii=False, default=str), getattr(self.__client, "api_key", None)).encode()
            request_ref = self.__host.artifacts.put(request)
            self.__host.emit("llm_start", identity=asdict(identity), sequence=sequence, request_ref=asdict(request_ref))
            started = time.perf_counter()
            try:
                response = self.__client.complete(messages, **kwargs)
            except Exception as exc:
                self.__host.emit("llm_failed", identity=asdict(identity), sequence=sequence, error=str(exc))
                raise
            duration = round((time.perf_counter() - started) * 1000, 2)
            response.call_identity = identity
            with self.__lock:
                self.usage["input_tokens"] += response.input_tokens
                self.usage["output_tokens"] += response.output_tokens
                self.usage["cached_tokens"] += response.cached_tokens
            output = {"content": response.content, "tool_calls": [asdict(tc) for tc in response.tool_calls],
                      "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
                      "cached_tokens": response.cached_tokens, "finish_reason": response.finish_reason}
            output_ref = self.__host.artifacts.put(redact_secret(json.dumps(output, ensure_ascii=False, default=str), getattr(self.__client, "api_key", None)).encode())
            self.__host.emit("llm_completed", identity=asdict(identity), sequence=sequence, duration_ms=duration,
                             response_ref=asdict(output_ref), input_tokens=response.input_tokens,
                             output_tokens=response.output_tokens, cached_tokens=response.cached_tokens)
            if self.trace:
                self.trace(identity, sequence, messages, response, duration)
            return response
