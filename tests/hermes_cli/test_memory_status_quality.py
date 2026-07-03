from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

from hermes_cli import memory_setup
from hermes_cli.subcommands.memory import build_memory_parser


def test_memory_status_parser_accepts_quality_json_flags():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_memory_parser(subparsers, cmd_memory=lambda args: None)

    args = parser.parse_args(["memory", "status", "--quality", "--json"])

    assert args.command == "memory"
    assert args.memory_command == "status"
    assert args.quality is True
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
