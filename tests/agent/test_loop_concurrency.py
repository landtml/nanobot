from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _provider() -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = SimpleNamespace(
        max_tokens=4096,
        temperature=0.1,
        reasoning_effort=None,
    )
    return provider


def test_request_concurrency_is_unlimited_by_default(
    monkeypatch: pytest.MonkeyPatch,
    loop_factory,
) -> None:
    monkeypatch.delenv("NANOBOT_MAX_CONCURRENT_REQUESTS", raising=False)

    loop = loop_factory(provider=_provider(), patch_deps=True)

    assert loop._concurrency_gate is None


def test_scheduler_default_and_legacy_turn_gate_keep_their_existing_parity(
    monkeypatch: pytest.MonkeyPatch,
    loop_factory,
) -> None:
    from nanobot.config.schema import Config
    from nanobot.orchestration.scheduler import ProviderLane
    from nanobot.providers.factory import _scheduler_for_config

    monkeypatch.delenv("NANOBOT_MAX_CONCURRENT_REQUESTS", raising=False)
    default_scheduler = _scheduler_for_config(Config())
    assert default_scheduler.limit_for(ProviderLane("provider", "model")) is None
    assert loop_factory(provider=_provider(), patch_deps=True)._concurrency_gate is None

    monkeypatch.setenv("NANOBOT_MAX_CONCURRENT_REQUESTS", "2")
    legacy_config = Config()
    mapped_scheduler = _scheduler_for_config(legacy_config)
    loop = loop_factory(provider=_provider(), patch_deps=True)

    assert mapped_scheduler.limit_for(ProviderLane("provider", "model")) == 2
    assert loop._concurrency_gate is not None
    assert loop._concurrency_gate._value == 2


@pytest.mark.asyncio
async def test_positive_request_concurrency_keeps_explicit_cap(
    monkeypatch: pytest.MonkeyPatch,
    loop_factory,
) -> None:
    monkeypatch.setenv("NANOBOT_MAX_CONCURRENT_REQUESTS", "2")
    loop = loop_factory(provider=_provider(), patch_deps=True)
    gate = loop._concurrency_gate

    assert gate is not None
    for _ in range(2):
        await gate.acquire()
    try:
        assert gate.locked()
    finally:
        for _ in range(2):
            gate.release()
