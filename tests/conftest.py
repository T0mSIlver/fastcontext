"""Shared pytest fixtures.

The LLM/agent tests exercise a real OpenAI-compatible endpoint. They are opt-in:
set ``FC_MODEL`` and ``FC_BASE_URL`` (legacy ``MODEL``/``BASE_URL`` also work) to
run them, otherwise they are skipped so the suite stays green offline.
"""

import os

import pytest


def _endpoint() -> dict[str, str | None]:
    return {
        "model": os.getenv("FC_MODEL") or os.getenv("MODEL"),
        "base_url": os.getenv("FC_BASE_URL") or os.getenv("BASE_URL"),
        "api_key": os.getenv("FC_API_KEY") or os.getenv("API_KEY"),
    }


@pytest.fixture
def llm_endpoint() -> dict[str, str | None]:
    """Endpoint config for live LLM tests; skips the test when none is set."""
    cfg = _endpoint()
    if not cfg["model"] or not cfg["base_url"]:
        pytest.skip("no LLM endpoint configured (set FC_MODEL and FC_BASE_URL to run live LLM tests)")
    return cfg
