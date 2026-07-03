from __future__ import annotations

from datetime import datetime, timezone

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class SnapshotProvider(MemoryProvider):
    def __init__(self) -> None:
        self.records = [
            {"id": "prov:1", "tier": "candidate", "content": "private duplicate fact"},
            {"id": "prov:2", "tier": "candidate", "content": " private duplicate fact "},
        ]
        self.observations = [
            {
                "expected_record_ids": ["prov:1", "prov:missing"],
                "retrieved_record_ids": ["prov:1", "prov:extra"],
                "query": "private recall query should never serialize",
            }
        ]

    @property
    def name(self) -> str:
        return "snapshot-test"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self):
        return []

    def quality_snapshot_records(self):
        return [dict(record) for record in self.records]

    def recall_snapshot_observations(self):
        return [dict(observation) for observation in self.observations]


def test_memory_manager_builds_provider_quality_report_without_mutating_provider_records():
    provider = SnapshotProvider()
    manager = MemoryManager()
    manager.add_provider(provider)

    records = manager.quality_snapshot_records()
    report = manager.build_quality_report(now=datetime(2026, 7, 3, 2, 40, tzinfo=timezone.utc))

    assert [record["id"] for record in records] == ["prov:1", "prov:2"]
    assert [record["source_provider"] for record in records] == ["snapshot-test", "snapshot-test"]
    assert provider.records == [
        {"id": "prov:1", "tier": "candidate", "content": "private duplicate fact"},
        {"id": "prov:2", "tier": "candidate", "content": " private duplicate fact "},
    ]

    serialized = report.to_dict()
    assert serialized["total_count"] == 2
    assert serialized["tier_counts"] == {"candidate": 2}
    assert serialized["duplicate_count"] == 1
    assert "private duplicate fact" not in repr(serialized)


def test_memory_manager_builds_recall_report_from_provider_observations_without_query_text():
    manager = MemoryManager()
    manager.add_provider(SnapshotProvider())

    observations = manager.recall_snapshot_observations()
    report = manager.build_recall_report()

    assert observations[0]["source_provider"] == "snapshot-test"
    serialized = report.to_dict()
    assert serialized["observation_count"] == 1
    assert serialized["expected_record_count"] == 2
    assert serialized["retrieved_record_count"] == 2
    assert serialized["hit_count"] == 1
    assert serialized["miss_count"] == 1
    assert serialized["unexpected_retrieval_count"] == 1
    assert {diag["reason"] for diag in serialized["diagnostics"]} == {
        "memory-recall-miss",
        "memory-recall-unexpected-retrieval",
    }
    assert "private recall query" not in repr(serialized)


def test_memory_manager_transition_report_compares_previous_snapshot_to_current_provider_state():
    manager = MemoryManager()
    manager.add_provider(SnapshotProvider())

    report = manager.build_transition_report(
        before_records=[
            {"id": "old:1", "tier": "stale", "content": "old private memory", "stale": True},
            {"id": "prov:1", "tier": "candidate", "content": "private duplicate fact"},
        ],
        events=[
            {
                "event_type": "promotion",
                "record_id": "prov:1",
                "content": "private duplicate fact",
            }
        ],
    )

    serialized = report.to_dict()
    assert serialized["before"]["total_count"] == 2
    assert serialized["after"]["total_count"] == 2
    assert serialized["stale_count_delta"] == -1
    assert serialized["duplicate_count_delta"] == 1
    assert serialized["event_counts"] == {"promotion": 1}
    assert "old private memory" not in repr(serialized)
    assert "private duplicate fact" not in repr(serialized)
