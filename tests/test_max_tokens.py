"""Hermetic tests for provider-driven max_tokens resolution (no network)."""

import pytest

from fastcontext.agent import llm as llm_module
from fastcontext.agent.llm import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    _scan_for_context_length,
    fetch_provider_context_length,
    resolve_max_completion_tokens,
    resolve_max_context,
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


def test_scan_llamacpp_nested_meta_prefers_the_served_window():
    """n_ctx (served) beats n_ctx_train (what the weights allow).

    This value now feeds --max-context, so it must describe THIS server. A model trained at 160000
    but served at 80000 can only hold 80000; believing the trained figure sets a budget twice the
    real window, so the budget never trips and the run dies with the overflow it exists to prevent.
    Under-reading only costs an early finalize; over-reading loses the answer.
    """
    entry = {"id": "m", "object": "model", "meta": {"n_ctx_train": 160000, "n_ctx": 80000}}
    assert _scan_for_context_length(entry) == 80000


def test_scan_falls_back_to_the_trained_window_when_served_is_absent():
    entry = {"id": "m", "object": "model", "meta": {"n_ctx_train": 160000}}
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


# --- fetch_provider_context_length -----------------------------------------------


def test_fetch_prefers_matching_model(monkeypatch):
    payload = {
        "data": [
            {"id": "other", "max_model_len": 4096},
            {"id": "target", "max_model_len": 128000},
        ]
    }
    _patch_models_response(monkeypatch, payload)
    assert fetch_provider_context_length("http://x/v1", model="target") == 128000


def test_fetch_falls_back_to_any_entry(monkeypatch):
    payload = {"data": [{"id": "only", "context_length": 8192}]}
    _patch_models_response(monkeypatch, payload)
    assert fetch_provider_context_length("http://x/v1", model="missing") == 8192


def test_fetch_returns_none_on_error(monkeypatch):
    _patch_models_response(monkeypatch, exc=RuntimeError("boom"))
    assert fetch_provider_context_length("http://x/v1", model="m") is None


def test_fetch_returns_none_without_base_url():
    assert fetch_provider_context_length("") is None


# --- resolve_max_completion_tokens -------------------------------------------
#
# The completion cap must never come from the provider. What a provider advertises is the context
# WINDOW; used as a per-response cap it makes the reserve (2 x this) exceed the window, so the agent
# finalizes before its first turn and answers with no exploration.


def test_completion_cap_explicit_int_wins():
    assert resolve_max_completion_tokens("2048") == 2048
    assert resolve_max_completion_tokens(1024) == 1024


def test_completion_cap_never_queries_the_provider(monkeypatch):
    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("the completion cap must not come from the provider")

    monkeypatch.setattr(llm_module, "fetch_provider_context_length", boom)
    assert resolve_max_completion_tokens(None) == DEFAULT_MAX_COMPLETION_TOKENS
    assert resolve_max_completion_tokens("auto") == DEFAULT_MAX_COMPLETION_TOKENS


def test_completion_cap_invalid_falls_back_to_default():
    assert resolve_max_completion_tokens("-5") == DEFAULT_MAX_COMPLETION_TOKENS
    assert resolve_max_completion_tokens("notanint") == DEFAULT_MAX_COMPLETION_TOKENS


def test_a_window_sized_completion_cap_would_disable_the_run():
    """Why the split exists, asserted rather than described.

    This is what the old `--max-tokens auto` produced against any provider that advertises: the
    budget limit goes negative and the agent finalizes before turn 1.
    """
    from fastcontext.agent.budget import ContextBudget, required_reserve

    window = 80_000  # what a provider reports (llama.cpp 160k ctx / 2 parallel)
    doomed = ContextBudget(max_context=72_000, reserve=required_reserve(12_000, window))
    assert doomed.must_finalize([], None), "premise: a window-sized completion cap kills the run"

    sane = ContextBudget(
        max_context=72_000, reserve=required_reserve(12_000, resolve_max_completion_tokens(None))
    )
    assert not sane.must_finalize([], None)


# --- resolve_max_context -----------------------------------------------------


def test_context_explicit_int_wins(monkeypatch):
    def boom(*a, **k):  # pragma: no cover - should never run
        raise AssertionError("provider should not be queried")

    monkeypatch.setattr(llm_module, "fetch_provider_context_length", boom)
    assert resolve_max_context("70000", base_url="http://x/v1") == 70000


def test_context_zero_disables_without_asking(monkeypatch):
    def boom(*a, **k):  # pragma: no cover - should never run
        raise AssertionError("provider should not be queried")

    monkeypatch.setattr(llm_module, "fetch_provider_context_length", boom)
    assert resolve_max_context("0", base_url="http://x/v1") == 0


def test_context_auto_fetches_from_provider(monkeypatch):
    monkeypatch.setattr(llm_module, "fetch_provider_context_length", lambda *a, **k: 65536)
    assert resolve_max_context("auto", base_url="http://x/v1") == 65536
    assert resolve_max_context(None, base_url="http://x/v1") == 65536


def test_context_falls_back_to_off_when_provider_says_nothing(monkeypatch):
    """Off, not a guess: too high and the run dies mid-flight, too low and it finalizes early."""
    monkeypatch.setattr(llm_module, "fetch_provider_context_length", lambda *a, **k: None)
    assert resolve_max_context("auto", base_url="http://x/v1") == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
