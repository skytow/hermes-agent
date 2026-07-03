from __future__ import annotations

from agent.memory_manager import MemoryManager
from plugins.memory.honcho import HonchoMemoryProvider
from typing import Any, cast


class _FakeHonchoManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get_peer_card(self, session_key: str, peer: str = "user") -> list[str]:
        self.calls.append((session_key, peer))
        if peer == "user":
            return [
                "Chris prefers concise, verification-backed status reports",
                " Chris prefers concise, verification-backed status reports ",
            ]
        if peer == "ai":
            return ["JR should keep live memory mutations disabled during audits"]
        return []


def _ready_provider(manager: _FakeHonchoManager | None = None) -> HonchoMemoryProvider:
    provider = HonchoMemoryProvider()
    cast(Any, provider)._manager = manager or _FakeHonchoManager()
    provider._session_key = "telegram:8703694071"
    provider._session_initialized = True
    return provider


def test_honcho_provider_quality_snapshot_exposes_peer_card_facts_without_raw_ids():
    provider = _ready_provider()

    records = provider.quality_snapshot_records()

    assert [(record["peer"], record["tier"], record["source"]) for record in records] == [
        ("user", "durable", "honcho-peer-card"),
        ("user", "durable", "honcho-peer-card"),
        ("ai", "durable", "honcho-peer-card"),
    ]
    assert records[0]["content"] == "Chris prefers concise, verification-backed status reports"
    assert all(record["id"].startswith("honcho:") for record in records)
    assert "Chris prefers" not in " ".join(record["id"] for record in records)

    manager = MemoryManager()
    manager.add_provider(provider)
    report = manager.build_quality_report().to_dict()

    assert report["total_count"] == 3
    assert report["duplicate_count"] == 1
    assert "Chris prefers concise" not in repr(report)
    assert "verification-backed" not in repr(report)


def test_honcho_provider_quality_snapshot_is_empty_before_session_ready():
    provider = HonchoMemoryProvider()

    assert provider.quality_snapshot_records() == []
