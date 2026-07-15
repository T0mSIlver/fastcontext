"""Hermetic tests for provider-driven max_tokens resolution (no network)."""

import pytest

from fastcontext.agent import llm as llm_module
from fastcontext.agent.llm import (
    DEFAULT_MAX_TOKENS,
    _scan_for_context_length,
    fetch_provider_max_tokens,
    resolve_max_tokens,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _patch_models_response(monkeypatch, payload=None, exc=None):
    def fake_get(url, headers=None, timeout=None):
        if exc is not None:
            raise exc
        return _FakeResponse(payload)

    monkeypatch.setattr(llm_module.httpx, "get", fake_get)


# --- _scan_for_context_length ------------------------------------------------


def test_scan_vllm_top_level():
    assert _scan_for_context_length({"id": "m", "max_model_len": 32768}) == 32768


def test_scan_llamacpp_nested_meta():
    entry = {"id": "m", "object": "model", "meta": {"n_ctx_train": 160000, "n_ctx": 80000}}
    # n_ctx_train is prioritised over n_ctx.
    assert _scan_for_context_length(entry) == 160000


def test_scan_llamacpp_router_args():
    # llama.cpp swapper exposes launch flags instead of a context field;
    # usable context = ctx-size // parallel.
    entry = {
        "id": "fastcontext",
        "status": {"args": ["/bin/llama-server", "--ctx-size", "160000", "--parallel", "2"]},
    }
    assert _scan_for_context_length(entry) == 80000


def test_scan_llamacpp_router_args_single_slot():
    entry = {"id": "m", "status": {"args": ["--ctx-size", "32768", "--parallel", "1"]}}
    assert _scan_for_context_length(entry) == 32768


def test_scan_ignores_bogus_values():
    assert _scan_for_context_length({"id": "m", "max_model_len": 0}) is None
    assert _scan_for_context_length({"id": "m", "context_length": "nan"}) is None
    assert _scan_for_context_length({"id": "m"}) is None


# --- fetch_provider_max_tokens -----------------------------------------------


def test_fetch_prefers_matching_model(monkeypatch):
    payload = {
        "data": [
            {"id": "other", "max_model_len": 4096},
            {"id": "target", "max_model_len": 128000},
        ]
    }
    _patch_models_response(monkeypatch, payload)
    assert fetch_provider_max_tokens("http://x/v1", model="target") == 128000


def test_fetch_falls_back_to_any_entry(monkeypatch):
    payload = {"data": [{"id": "only", "context_length": 8192}]}
    _patch_models_response(monkeypatch, payload)
    assert fetch_provider_max_tokens("http://x/v1", model="missing") == 8192


def test_fetch_returns_none_on_error(monkeypatch):
    _patch_models_response(monkeypatch, exc=RuntimeError("boom"))
    assert fetch_provider_max_tokens("http://x/v1", model="m") is None


def test_fetch_returns_none_without_base_url():
    assert fetch_provider_max_tokens("") is None


# --- resolve_max_tokens ------------------------------------------------------


def test_resolve_explicit_int_wins(monkeypatch):
    # Explicit value must not trigger a provider call.
    def boom(*a, **k):  # pragma: no cover - should never run
        raise AssertionError("provider should not be queried")

    monkeypatch.setattr(llm_module, "fetch_provider_max_tokens", boom)
    assert resolve_max_tokens("2048", base_url="http://x/v1") == 2048
    assert resolve_max_tokens(1024, base_url="http://x/v1") == 1024


def test_resolve_auto_fetches_from_provider(monkeypatch):
    monkeypatch.setattr(llm_module, "fetch_provider_max_tokens", lambda *a, **k: 65536)
    assert resolve_max_tokens("auto", base_url="http://x/v1") == 65536
    assert resolve_max_tokens(None, base_url="http://x/v1") == 65536


def test_resolve_falls_back_to_default(monkeypatch):
    monkeypatch.setattr(llm_module, "fetch_provider_max_tokens", lambda *a, **k: None)
    assert resolve_max_tokens("auto", base_url="http://x/v1") == DEFAULT_MAX_TOKENS


def test_resolve_invalid_explicit_falls_through_to_provider(monkeypatch):
    monkeypatch.setattr(llm_module, "fetch_provider_max_tokens", lambda *a, **k: 12345)
    assert resolve_max_tokens("-5", base_url="http://x/v1") == 12345
    assert resolve_max_tokens("notanint", base_url="http://x/v1") == 12345


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
