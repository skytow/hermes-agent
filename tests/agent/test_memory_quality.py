from __future__ import annotations

from datetime import datetime, timezone

from agent.memory_quality import (
    build_memory_quality_recall_report,
    build_memory_quality_report,
    build_memory_quality_transition_report,
)
from tools.memory_tool import MemoryStore


def test_memory_quality_report_counts_tiers_and_quality_signals():
    report = build_memory_quality_report(
        [
            {
                "id": "mem-a",
                "tier": "durable",
                "content": "Chris prefers concise updates.",
                "confidence": 0.9,
            },
            {
                "id": "mem-b",
                "tier": "durable",
                "content": "  Chris prefers concise updates.  ",
                "confidence": 0.8,
            },
            {
                "id": "mem-c",
                "tier": "candidate",
                "content": "Temporary inferred preference.",
                "confidence": 0.2,
                "stale": True,
            },
            {
                "id": "mem-d",
                "tier": "conflicted",
                "content": "Chris prefers long-form updates.",
                "confidence": 0.6,
                "conflict_status": "unresolved",
            },
        ],
        now=datetime(2026, 7, 3, 0, 0, tzinfo=timezone.utc),
        obsidian_synced_at=datetime(2026, 7, 2, 23, 0, tzinfo=timezone.utc),
        queued_write_count=3,
    )

    assert report.tier_counts == {
        "candidate": 1,
        "conflicted": 1,
        "durable": 2,
    }
    assert report.total_count == 4
    assert report.duplicate_count == 1
    assert report.duplicate_rate == 0.25
    assert report.stale_count == 1
    assert report.stale_rate == 0.25
    assert report.unresolved_conflict_count == 1
    assert report.average_confidence == 0.625
    assert report.obsidian_sync_lag_seconds == 3600
    assert report.queued_write_count == 3


def test_memory_quality_diagnostics_explain_actions_without_private_content():
    report = build_memory_quality_report(
        [
            {"id": "first", "tier": "durable", "content": "same sensitive fact", "confidence": 0.9},
            {"id": "dupe", "tier": "durable", "content": "same sensitive fact", "confidence": 0.9},
            {"id": "stale", "tier": "stale", "content": "old private fact", "confidence": 0.4},
            {
                "id": "conflict",
                "tier": "durable",
                "content": "conflicting private fact",
                "confidence": 0.5,
                "conflict": True,
            },
        ]
    )

    serialized = report.to_dict()
    reasons = {diag["reason"] for diag in serialized["diagnostics"]}
    assert reasons == {
        "exact-duplicate-merge-candidate",
        "stale-memory-review-needed",
        "unresolved-conflict-review-needed",
    }

    duplicate = next(
        diag for diag in serialized["diagnostics"]
        if diag["reason"] == "exact-duplicate-merge-candidate"
    )
    assert duplicate["record_ids"] == ["first", "dupe"]
    assert duplicate["canonical_record_id"] == "first"
    assert duplicate["content_fingerprint"]

    # Diagnostics are audit-safe: they include ids/reasons/fingerprints, not raw memory text.
    assert "same sensitive fact" not in repr(serialized)
    assert "old private fact" not in repr(serialized)
    assert "conflicting private fact" not in repr(serialized)


def test_memory_store_builds_audit_safe_quality_report_from_live_snapshot():
    store = MemoryStore()
    store.memory_entries = ["Duplicate private fact", " duplicate private fact "]
    store.user_entries = ["User durable preference"]

    records = store.quality_snapshot_records()
    report = store.build_quality_report(
        now=datetime(2026, 7, 3, 0, 30, tzinfo=timezone.utc),
        obsidian_synced_at="2026-07-03T00:00:00Z",
        queued_write_count=2,
    )

    assert [record["id"] for record in records] == ["memory:0", "memory:1", "user:0"]
    assert [record["tier"] for record in records] == ["durable", "durable", "durable"]
    assert store.memory_entries == ["Duplicate private fact", " duplicate private fact "]
    assert store.user_entries == ["User durable preference"]

    serialized = report.to_dict()
    assert serialized["total_count"] == 3
    assert serialized["tier_counts"] == {"durable": 3}
    assert serialized["duplicate_count"] == 1
    assert serialized["queued_write_count"] == 2
    assert serialized["obsidian_sync_lag_seconds"] == 1800
    assert "Duplicate private fact" not in repr(serialized)
    assert "User durable preference" not in repr(serialized)


def test_memory_quality_transition_report_tracks_gc_event_deltas_without_private_content():
    report = build_memory_quality_transition_report(
        before_records=[
            {"id": "candidate-a", "tier": "candidate", "content": "sensitive duplicate fact"},
            {"id": "candidate-b", "tier": "candidate", "content": " sensitive duplicate fact "},
            {"id": "stale-c", "tier": "stale", "content": "old private detail", "stale": True},
            {
                "id": "conflict-d",
                "tier": "conflicted",
                "content": "conflicting private detail",
                "conflict_status": "unresolved",
            },
        ],
        after_records=[
            {"id": "candidate-a", "tier": "durable", "content": "sensitive duplicate fact"},
            {"id": "conflict-d", "tier": "durable", "content": "conflicting private detail"},
        ],
        events=[
            {"event_type": "promotion", "record_id": "candidate-a", "content": "sensitive duplicate fact"},
            {"event_type": "merge", "record_ids": ["candidate-a", "candidate-b"], "canonical_record_id": "candidate-a"},
            {"event_type": "deletion", "record_id": "stale-c", "reason": "expired-stale"},
            {"event_type": "conflict_resolution", "record_id": "conflict-d"},
            {"event_type": "obsidian_sync", "record_id": "candidate-a"},
            {"event_type": "local_index_rebuild", "record_id": "candidate-a"},
        ],
    )

    serialized = report.to_dict()

    assert serialized["before"]["total_count"] == 4
    assert serialized["after"]["total_count"] == 2
    assert serialized["total_count_delta"] == -2
    assert serialized["duplicate_count_delta"] == -1
    assert serialized["stale_count_delta"] == -1
    assert serialized["unresolved_conflict_count_delta"] == -1
    assert serialized["tier_count_delta"] == {
        "candidate": -2,
        "conflicted": -1,
        "durable": 2,
        "stale": -1,
    }
    assert serialized["event_counts"] == {
        "conflict_resolution": 1,
        "deletion": 1,
        "local_index_rebuild": 1,
        "merge": 1,
        "obsidian_sync": 1,
        "promotion": 1,
    }
    assert {diag["reason"] for diag in serialized["event_diagnostics"]} == {
        "memory-event-conflict-resolution",
        "memory-event-deletion",
        "memory-event-local-index-rebuild",
        "memory-event-merge",
        "memory-event-obsidian-sync",
        "memory-event-promotion",
    }
    promotion = next(diag for diag in serialized["event_diagnostics"] if diag["reason"] == "memory-event-promotion")
    assert "content_fingerprint" in promotion
    assert "sensitive duplicate fact" not in repr(serialized)
    assert "old private detail" not in repr(serialized)
    assert "conflicting private detail" not in repr(serialized)


def test_memory_quality_recall_report_tracks_hits_and_misses_without_query_text():
    report = build_memory_quality_recall_report(
        observations=[
            {
                "id": "obs-1",
                "expected_record_ids": ["memory:0", "user:0"],
                "retrieved_record_ids": ["memory:0", "other:1"],
                "query": "private customer name and account detail",
            },
            {
                "id": "obs-2",
                "expected_record_id": "memory:2",
                "retrieved_record_ids": [],
                "query_text": "another private lookup",
            },
            {
                "id": "obs-3",
                "expected_record_ids": [],
                "retrieved_record_ids": ["memory:3"],
                "query": "private exploratory query",
            },
        ]
    )

    serialized = report.to_dict()

    assert serialized["observation_count"] == 3
    assert serialized["expected_record_count"] == 3
    assert serialized["retrieved_record_count"] == 3
    assert serialized["hit_count"] == 1
    assert serialized["miss_count"] == 2
    assert serialized["unexpected_retrieval_count"] == 2
    assert serialized["recall_rate"] == 1 / 3
    assert serialized["precision_rate"] == 1 / 3
    reasons = {diag["reason"] for diag in serialized["diagnostics"]}
    assert reasons == {
        "memory-recall-miss",
        "memory-recall-unexpected-retrieval",
    }
    miss = next(diag for diag in serialized["diagnostics"] if diag["reason"] == "memory-recall-miss")
    assert miss["record_ids"] == ["user:0", "memory:2"]
    assert "private customer name" not in repr(serialized)
    assert "another private lookup" not in repr(serialized)
    assert "private exploratory query" not in repr(serialized)
