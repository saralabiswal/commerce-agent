"""Tests for local tracing behavior and observability safeguards."""

from observability.tracing import _has_real_langsmith_key


def test_placeholder_langsmith_key_disables_remote_tracing():
    """Example keys in .env should not trigger LangSmith uploads."""
    assert _has_real_langsmith_key("") is False
    assert _has_real_langsmith_key("your_langsmith_api_key_here") is False
    assert _has_real_langsmith_key("placeholder") is False
    assert _has_real_langsmith_key("lsv2_pt_realistic_token") is True
