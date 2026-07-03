from __future__ import annotations

from agent.memory_manager import MemoryManager
from plugins.memory.honcho import HonchoMemoryProvider
from typing import Any, cast


class _FakeHonchoManager:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

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


class _FakeRecallManager(_FakeHonchoManager):
    def get_peer_card(self, session_key: str, peer: str = "user") -> list[str]:
        self.calls.append((session_key, peer))
        if peer == "user":
            return [
                "Chris wants status reports backed by commands",
                "Chris prioritizes direct revenue before cleanup",
            ]
        return []

    def search_context(
        self,
        session_key: str,
        query: str,
        max_tokens: int = 800,
        peer: str = "user",
    ) -> str:
        return "Relevant context: Chris wants status reports backed by commands."

    def get_session_context(self, session_key: str, peer: str = "user") -> dict[str, Any]:
        return {
            "card": "Chris prioritizes direct revenue before cleanup",
            "summary": "Private session summary should not become recall evidence text",
        }


class _FakeTransitionManager(_FakeRecallManager):
    def create_conclusion(self, session_key: str, conclusion: str, peer: str = "user") -> bool:
        self.calls.append(("create_conclusion", session_key, peer, conclusion))
        return True

    def delete_conclusion(self, session_key: str, delete_id: str, peer: str = "user") -> bool:
        self.calls.append(("delete_conclusion", session_key, peer, delete_id))
        return True


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


def test_honcho_provider_recall_snapshot_tracks_search_and_context_hits_without_raw_text():
    provider = _ready_provider(_FakeRecallManager())

    search_payload = provider.handle_tool_call(
        "honcho_search",
        {"query": "private customer status report query", "peer": "user"},
    )
    context_payload = provider.handle_tool_call("honcho_context", {"peer": "user"})

    assert "Chris wants status reports" in search_payload
    assert "direct revenue" in context_payload

    observations = provider.recall_snapshot_observations()
    assert [observation["route"] for observation in observations] == [
        "honcho_search",
        "honcho_context",
    ]
    assert all("query" not in observation for observation in observations)
    assert all("private customer" not in repr(observation) for observation in observations)
    assert observations[0]["expected_record_ids"]
    assert observations[0]["retrieved_record_ids"] == [observations[0]["expected_record_ids"][0]]
    assert observations[1]["retrieved_record_ids"] == [observations[1]["expected_record_ids"][1]]

    manager = MemoryManager()
    manager.add_provider(provider)
    report = manager.build_recall_report().to_dict()

    assert report["observation_count"] == 2
    assert report["expected_record_count"] == 4
    assert report["retrieved_record_count"] == 2
    assert report["hit_count"] == 2
    assert report["miss_count"] == 2
    assert "private customer status report query" not in repr(report)
    assert "Chris wants status reports" not in repr(report)
    assert "direct revenue" not in repr(report)


def test_honcho_provider_transition_events_track_conclusion_writes_without_raw_text():
    provider = _ready_provider(_FakeTransitionManager())

    create_payload = provider.handle_tool_call(
        "honcho_conclude",
        {"conclusion": "Private payment detail that must not serialize", "peer": "user"},
    )
    delete_payload = provider.handle_tool_call(
        "honcho_conclude",
        {"delete_id": "conclusion-secret-123", "peer": "user"},
    )

    assert "Conclusion saved" in create_payload
    assert "Conclusion conclusion-secret-123 deleted" in delete_payload

    events = provider.transition_snapshot_events()
    assert [event["event_type"] for event in events] == [
        "conclusion_create",
        "conclusion_delete",
    ]
    assert all(event["record_id"].startswith("honcho:user:conclusion:") for event in events)
    assert "Private payment detail" not in repr(events)
    assert "conclusion-secret-123" not in repr(events)

    manager = MemoryManager()
    manager.add_provider(provider)
    report = manager.build_transition_report(before_records=[]).to_dict()

    assert report["event_counts"] == {"conclusion_create": 1, "conclusion_delete": 1}
    assert "Private payment detail" not in repr(report)
    assert "conclusion-secret-123" not in repr(report)
