from __future__ import annotations

from datetime import datetime, timezone

from agent.memory_quality import build_memory_quality_report


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
