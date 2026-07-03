"""Side-effect-safe memory quality metrics and diagnostics.

This module intentionally does not read, write, merge, demote, archive, or delete
memory records.  It converts already-snapshotted memory/provider records into a
small audit-safe report that callers can persist or render elsewhere.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


_TIER_ALIASES = {
    "": "unknown",
    "memory": "durable",
    "user": "durable",
}


@dataclass(frozen=True)
class MemoryQualityDiagnostic:
    """Audit-safe explanation for one memory quality signal."""

    reason: str
    record_ids: list[str]
    severity: str = "info"
    canonical_record_id: str | None = None
    content_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "reason": self.reason,
            "severity": self.severity,
            "record_ids": list(self.record_ids),
        }
        if self.canonical_record_id is not None:
            data["canonical_record_id"] = self.canonical_record_id
        if self.content_fingerprint is not None:
            data["content_fingerprint"] = self.content_fingerprint
        return data


@dataclass(frozen=True)
class MemoryQualityReport:
    """Aggregated memory quality counters plus audit-safe diagnostics."""

    total_count: int
    tier_counts: dict[str, int]
    duplicate_count: int
    duplicate_rate: float
    stale_count: int
    stale_rate: float
    unresolved_conflict_count: int
    average_confidence: float | None
    obsidian_sync_lag_seconds: int | None = None
    queued_write_count: int = 0
    diagnostics: list[MemoryQualityDiagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_count": self.total_count,
            "tier_counts": dict(self.tier_counts),
            "duplicate_count": self.duplicate_count,
            "duplicate_rate": self.duplicate_rate,
            "stale_count": self.stale_count,
            "stale_rate": self.stale_rate,
            "unresolved_conflict_count": self.unresolved_conflict_count,
            "average_confidence": self.average_confidence,
            "obsidian_sync_lag_seconds": self.obsidian_sync_lag_seconds,
            "queued_write_count": self.queued_write_count,
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
        }


@dataclass(frozen=True)
class MemoryQualityTransitionReport:
    """Before/after memory quality deltas plus audit-safe event evidence."""

    before: MemoryQualityReport
    after: MemoryQualityReport
    total_count_delta: int
    tier_count_delta: dict[str, int]
    duplicate_count_delta: int
    stale_count_delta: int
    unresolved_conflict_count_delta: int
    event_counts: dict[str, int]
    event_diagnostics: list[MemoryQualityDiagnostic] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "total_count_delta": self.total_count_delta,
            "tier_count_delta": dict(self.tier_count_delta),
            "duplicate_count_delta": self.duplicate_count_delta,
            "stale_count_delta": self.stale_count_delta,
            "unresolved_conflict_count_delta": self.unresolved_conflict_count_delta,
            "event_counts": dict(self.event_counts),
            "event_diagnostics": [diagnostic.to_dict() for diagnostic in self.event_diagnostics],
        }


def build_memory_quality_report(
    records: Iterable[Mapping[str, Any]],
    *,
    now: datetime | str | None = None,
    obsidian_synced_at: datetime | str | None = None,
    queued_write_count: int = 0,
) -> MemoryQualityReport:
    """Build memory quality metrics from already-collected records.

    ``records`` may come from built-in memory, a provider snapshot, Obsidian
    note hydration, a brain index, or a candidate queue.  The helper only reads
    mapping values and returns counters/diagnostics; it never mutates source
    records and never includes raw memory content in the report.
    """

    normalized_records = [_normalize_record(record, index) for index, record in enumerate(records)]
    total = len(normalized_records)
    tier_counts = dict(sorted(Counter(record["tier"] for record in normalized_records).items()))

    stale_records = [record for record in normalized_records if record["is_stale"]]
    conflict_records = [record for record in normalized_records if record["has_unresolved_conflict"]]
    confidences = [record["confidence"] for record in normalized_records if record["confidence"] is not None]

    duplicate_count = 0
    diagnostics: list[MemoryQualityDiagnostic] = []
    duplicate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in normalized_records:
        if record["content_key"]:
            duplicate_groups[record["content_key"]].append(record)

    for content_key, group in duplicate_groups.items():
        if len(group) < 2:
            continue
        duplicate_count += len(group) - 1
        diagnostics.append(
            MemoryQualityDiagnostic(
                reason="exact-duplicate-merge-candidate",
                severity="warning",
                record_ids=[record["id"] for record in group],
                canonical_record_id=group[0]["id"],
                content_fingerprint=_fingerprint(content_key),
            )
        )

    for record in stale_records:
        diagnostics.append(
            MemoryQualityDiagnostic(
                reason="stale-memory-review-needed",
                severity="warning",
                record_ids=[record["id"]],
            )
        )

    for record in conflict_records:
        diagnostics.append(
            MemoryQualityDiagnostic(
                reason="unresolved-conflict-review-needed",
                severity="error",
                record_ids=[record["id"]],
            )
        )

    lag_seconds = _sync_lag_seconds(now=now, obsidian_synced_at=obsidian_synced_at)

    return MemoryQualityReport(
        total_count=total,
        tier_counts=tier_counts,
        duplicate_count=duplicate_count,
        duplicate_rate=_rate(duplicate_count, total),
        stale_count=len(stale_records),
        stale_rate=_rate(len(stale_records), total),
        unresolved_conflict_count=len(conflict_records),
        average_confidence=(sum(confidences) / len(confidences) if confidences else None),
        obsidian_sync_lag_seconds=lag_seconds,
        queued_write_count=max(0, int(queued_write_count)),
        diagnostics=diagnostics,
    )


def build_memory_quality_transition_report(
    *,
    before_records: Iterable[Mapping[str, Any]],
    after_records: Iterable[Mapping[str, Any]],
    events: Iterable[Mapping[str, Any]] | None = None,
    now: datetime | str | None = None,
    before_obsidian_synced_at: datetime | str | None = None,
    after_obsidian_synced_at: datetime | str | None = None,
    before_queued_write_count: int = 0,
    after_queued_write_count: int = 0,
) -> MemoryQualityTransitionReport:
    """Compare two already-snapshotted memory quality states.

    The transition report is intended for garbage-collection/refinement audit
    surfaces.  It tracks before/after counters and compact event reasons for
    promotions, deletions, merges, conflict resolution, Obsidian sync, local
    index rebuilds, or any future event type supplied by a scheduler.  It does
    not mutate source records and it never serializes raw memory contents.
    """

    before = build_memory_quality_report(
        before_records,
        now=now,
        obsidian_synced_at=before_obsidian_synced_at,
        queued_write_count=before_queued_write_count,
    )
    after = build_memory_quality_report(
        after_records,
        now=now,
        obsidian_synced_at=after_obsidian_synced_at,
        queued_write_count=after_queued_write_count,
    )
    event_counts, event_diagnostics = _event_metrics(events or [])
    return MemoryQualityTransitionReport(
        before=before,
        after=after,
        total_count_delta=after.total_count - before.total_count,
        tier_count_delta=_count_delta(before.tier_counts, after.tier_counts),
        duplicate_count_delta=after.duplicate_count - before.duplicate_count,
        stale_count_delta=after.stale_count - before.stale_count,
        unresolved_conflict_count_delta=(
            after.unresolved_conflict_count - before.unresolved_conflict_count
        ),
        event_counts=event_counts,
        event_diagnostics=event_diagnostics,
    )


def _normalize_record(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    record_id = str(record.get("id") or record.get("key") or f"record-{index}")
    tier = _normalize_tier(record.get("tier") or record.get("state") or record.get("target"))
    content = str(record.get("content") or record.get("text") or record.get("value") or "")
    confidence = _parse_confidence(record.get("confidence"))
    conflict_status = str(record.get("conflict_status") or record.get("conflictStatus") or "").lower()
    return {
        "id": record_id,
        "tier": tier,
        "content_key": _content_key(content),
        "confidence": confidence,
        "is_stale": bool(record.get("stale")) or tier == "stale",
        "has_unresolved_conflict": bool(record.get("conflict"))
        or tier in {"conflicted", "conflict"}
        or conflict_status in {"unresolved", "conflicted", "conflict", "unresolved_conflict"},
    }


def _count_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    tiers = sorted(set(before) | set(after))
    return {
        tier: delta
        for tier in tiers
        if (delta := int(after.get(tier, 0)) - int(before.get(tier, 0))) != 0
    }


def _event_metrics(
    events: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, int], list[MemoryQualityDiagnostic]]:
    counts: Counter[str] = Counter()
    diagnostics: list[MemoryQualityDiagnostic] = []
    for event in events:
        event_type = _normalize_event_type(
            event.get("event_type") or event.get("eventType") or event.get("action")
        )
        counts[event_type] += 1
        content = str(event.get("content") or event.get("text") or event.get("value") or "")
        content_key = _content_key(content)
        diagnostics.append(
            MemoryQualityDiagnostic(
                reason=f"memory-event-{event_type.replace('_', '-')}",
                severity="info",
                record_ids=_event_record_ids(event),
                canonical_record_id=_event_canonical_record_id(event),
                content_fingerprint=(_fingerprint(content_key) if content_key else None),
            )
        )
    return dict(sorted(counts.items())), diagnostics


def _event_record_ids(event: Mapping[str, Any]) -> list[str]:
    raw_ids = event.get("record_ids") or event.get("recordIds")
    if raw_ids is None:
        raw_id = event.get("record_id") or event.get("recordId") or event.get("id") or event.get("key")
        raw_ids = [raw_id] if raw_id is not None else []
    if isinstance(raw_ids, (str, bytes)):
        raw_ids = [raw_ids]
    try:
        return [str(record_id) for record_id in raw_ids if record_id is not None]
    except TypeError:
        return [str(raw_ids)] if raw_ids is not None else []


def _event_canonical_record_id(event: Mapping[str, Any]) -> str | None:
    value = event.get("canonical_record_id") or event.get("canonicalRecordId")
    if value is None:
        return None
    return str(value)


def _normalize_event_type(value: Any) -> str:
    event_type = str(value or "unknown").strip().lower().replace(" ", "_").replace("-", "_")
    return event_type or "unknown"


def _normalize_tier(value: Any) -> str:
    tier = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    return _TIER_ALIASES.get(tier, tier or "unknown")


def _parse_confidence(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence < 0:
        return 0.0
    if confidence > 1:
        return 1.0
    return confidence


def _content_key(content: str) -> str:
    return " ".join(content.casefold().split())


def _fingerprint(content_key: str) -> str:
    return hashlib.sha256(content_key.encode("utf-8")).hexdigest()[:12]


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _sync_lag_seconds(
    *,
    now: datetime | str | None,
    obsidian_synced_at: datetime | str | None,
) -> int | None:
    if obsidian_synced_at is None:
        return None
    end = _parse_datetime(now) or datetime.now(timezone.utc)
    synced = _parse_datetime(obsidian_synced_at)
    if synced is None:
        return None
    return max(0, int((end - synced).total_seconds()))


def _parse_datetime(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
