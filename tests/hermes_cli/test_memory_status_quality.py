from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

from agent.memory_provider import MemoryProvider
from hermes_cli import memory_setup
from hermes_cli.subcommands.memory import build_memory_parser


class CliSnapshotProvider(MemoryProvider):
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
        return [
            {"id": "provider:1", "tier": "candidate", "content": "private provider fact"},
            {"id": "provider:2", "tier": "candidate", "content": " private provider fact "},
        ]

    def recall_snapshot_observations(self):
        return [
            {
                "expected_record_ids": ["provider:1", "provider:missing"],
                "retrieved_record_ids": ["provider:1", "provider:extra"],
                "query": "private provider query should not serialize",
            }
        ]


def test_memory_status_parser_accepts_quality_json_flags():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_memory_parser(subparsers, cmd_memory=lambda args: None)

    args = parser.parse_args([
        "memory",
        "status",
        "--quality",
        "--provider-quality",
        "--quality-output",
        "/tmp/memory-health.json",
        "--json",
    ])

    assert args.command == "memory"
    assert args.memory_command == "status"
    assert args.quality is True
    assert args.provider_quality is True
    assert args.quality_output == "/tmp/memory-health.json"
    assert args.json is True


def test_memory_status_quality_json_is_audit_safe(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / ".hermes"
    memories = hermes_home / "memories"
    memories.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [])

    (memories / "MEMORY.md").write_text(
        "Private duplicate fact\n§\n private duplicate fact ",
        encoding="utf-8",
    )
    (memories / "USER.md").write_text("Sensitive user preference", encoding="utf-8")

    memory_setup.cmd_status(SimpleNamespace(quality=True, json=True))

    output = capsys.readouterr().out
    payload = json.loads(output)
    report = payload["quality_report"]

    assert payload["built_in"] == "always active"
    assert payload["provider"] is None
    assert report["total_count"] == 3
    assert report["tier_counts"] == {"durable": 3}
    assert report["duplicate_count"] == 1
    assert report["diagnostics"][0]["reason"] == "exact-duplicate-merge-candidate"
    assert "content_fingerprint" in report["diagnostics"][0]
    assert "Private duplicate fact" not in output
    assert "Sensitive user preference" not in output


def test_memory_status_provider_quality_payload_is_audit_safe(monkeypatch):
    monkeypatch.setattr(
        memory_setup,
        "_load_memory_provider_for_quality",
        lambda provider_name: CliSnapshotProvider(),
    )

    payload = memory_setup._memory_status_payload(
        provider_name="snapshot-test",
        include_quality=False,
        include_provider_quality=True,
    )

    provider_quality = payload["provider_quality"]
    assert provider_quality["provider"] == "snapshot-test"
    assert provider_quality["available"] is True
    assert provider_quality["quality_report"]["total_count"] == 2
    assert provider_quality["quality_report"]["duplicate_count"] == 1
    assert provider_quality["recall_report"]["observation_count"] == 1
    assert provider_quality["recall_report"]["miss_count"] == 1
    assert "private provider fact" not in repr(payload)
    assert "private provider query" not in repr(payload)


def test_memory_status_quality_output_writes_audit_safe_file(tmp_path, monkeypatch, capsys):
    hermes_home = tmp_path / ".hermes"
    memories = hermes_home / "memories"
    memories.mkdir(parents=True)
    output_path = tmp_path / "diagnostics" / "memory-health.json"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(memory_setup, "_get_available_providers", lambda: [])

    (memories / "MEMORY.md").write_text(
        "Private duplicate fact\n§\n private duplicate fact ",
        encoding="utf-8",
    )
    (memories / "USER.md").write_text("Sensitive user preference", encoding="utf-8")

    memory_setup.cmd_status(
        SimpleNamespace(
            quality=True,
            json=False,
            provider_quality=False,
            quality_output=str(output_path),
        )
    )

    output = capsys.readouterr().out
    written = output_path.read_text(encoding="utf-8")
    payload = json.loads(written)

    assert "Wrote audit-safe memory quality JSON" in output
    assert payload["quality_report"]["duplicate_count"] == 1
    assert "Private duplicate fact" not in written
    assert "Sensitive user preference" not in written
