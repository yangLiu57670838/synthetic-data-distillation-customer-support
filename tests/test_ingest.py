"""Offline tests for FR-1 seed ingestion."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data.ingest import run_ingest
from data.schema import BITEXT_REQUIRED_COLUMNS


def _write_mini_csv(path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "flags": "B",
                "instruction": "I want a refund for my broken item",
                "category": "REFUND",
                "intent": "get_refund",
                "response": "I can help with a refund request.",
            },
            {
                "flags": "Q",
                "instruction": "where is my package",
                "category": "ORDER",
                "intent": "track_order",
                "response": "I can help you track your order.",
            },
            {
                "flags": "",
                "instruction": "",  # invalid → dropped
                "category": "ORDER",
                "intent": "place_order",
                "response": "Sure.",
            },
        ]
    )
    assert BITEXT_REQUIRED_COLUMNS.issubset(frame.columns)
    frame.to_csv(path, index=False)


def test_run_ingest_local_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "mini_bitext.csv"
    _write_mini_csv(csv_path)

    config = {
        "source": {
            "hf_dataset_id": "unused/local-override",
            "hf_split": "train",
            "hf_revision": None,
            "local_path": str(csv_path),
        },
        "output": {
            "normalized_path": str(tmp_path / "seed_normalized.jsonl"),
            "report_path": str(tmp_path / "seed_quality_report.json"),
            "manifest_path": str(tmp_path / "seed_manifest.json"),
        },
        "licence_note": "test licence",
        "provenance_note": "test provenance",
    }

    report = run_ingest(config)

    assert report.n_rows_raw == 3
    assert report.n_rows_normalized == 2
    assert report.n_rows_dropped == 1
    assert report.intent_counts == {"get_refund": 1, "track_order": 1}

    normalized = Path(config["output"]["normalized_path"])
    lines = normalized.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["seed_id"] == "seed_000001"
    assert first["user_message"] == "I want a refund for my broken item"
    assert first["assistant_response"] == "I can help with a refund request."

    manifest = json.loads(Path(config["output"]["manifest_path"]).read_text())
    assert manifest["n_rows"] == 2
    assert len(manifest["normalized_sha256"]) == 64
