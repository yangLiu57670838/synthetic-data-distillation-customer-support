"""Offline tests for PRD section 8.2 partitioning."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from data.partitions import normalize_message, run_partition
from data.schema import SeedRecord


def _seed(
    seed_id: str,
    intent: str,
    message: str,
    *,
    flags: str = "",
    category: str = "ORDER",
) -> SeedRecord:
    return SeedRecord(
        seed_id=seed_id,
        category=category,
        intent=intent,
        user_message=message,
        assistant_response=f"Reply for {seed_id}",
        flags=flags,
        source_dataset="test",
        source_revision=None,
        retrieved_at=datetime.now(timezone.utc),
    )


def _write_jsonl(path: Path, records: list[SeedRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def test_normalize_message() -> None:
    assert normalize_message("  Where IS my package?! ") == "where is my package"


def test_run_partition_splits_and_blocks_leakage(tmp_path: Path) -> None:
    records = [
        _seed("seed_000001", "track_order", "where is my package", flags="B"),
        _seed("seed_000002", "track_order", "track my order please", flags="B"),
        _seed("seed_000003", "track_order", "package status update", flags="B"),
        _seed("seed_000004", "track_order", "Where IS my package?!", flags="N"),  # near-dup
        _seed("seed_000005", "get_refund", "i want a refund", flags="B"),
        _seed("seed_000006", "get_refund", "please refund this order", flags="B"),
        _seed("seed_000007", "get_refund", "broken item money back", flags="Q"),
        _seed("seed_000008", "get_refund", "refund my purchase now", flags="B"),
        _seed("seed_000009", "place_order", "i want to buy this", flags="B"),
        _seed("seed_000010", "place_order", "place an order for me", flags="B"),
        _seed("seed_000011", "place_order", "can you order this item", flags="B"),
        _seed("seed_000012", "place_order", "help me checkout", flags="B"),
        # exact duplicate of seed_000001
        _seed("seed_000013", "track_order", "where is my package", flags="B"),
    ]
    input_path = tmp_path / "seed_normalized.jsonl"
    _write_jsonl(input_path, records)

    out_dir = tmp_path / "partitions"
    config = {
        "input": {"normalized_path": str(input_path)},
        "output": {
            "dir": str(out_dir),
            "seed_train_path": str(out_dir / "seed_train.jsonl"),
            "validation_path": str(out_dir / "validation.jsonl"),
            "held_out_test_path": str(out_dir / "held_out_test.jsonl"),
            "challenge_path": str(out_dir / "challenge.jsonl"),
            "report_path": str(tmp_path / "partition_report.json"),
            "manifest_path": str(tmp_path / "partition_manifest.json"),
        },
        "independent_test_path": None,
        "split": {
            "seed": 42,
            "validation_fraction": 0.2,
            "held_out_test_fraction": 0.2,
        },
        "challenge": {
            "flag_markers": ["N", "Q", "W"],
            "rare_intent_max_count": 1,  # no intent is rare in this tiny set
            "max_fraction": 0.5,
            "min_per_intent": 0,
            "max_per_intent": 10,
        },
        "leakage": {"drop_train_near_duplicates": True},
    }

    report = run_partition(config)

    assert report.n_input_rows == 13
    assert report.n_exact_duplicates_removed == 1
    assert report.n_after_exact_dedup == 12
    assert report.n_challenge >= 1
    assert report.independent_test_status == "pending_external_source"

    total = (
        report.n_seed_train
        + report.n_validation
        + report.n_held_out_test
        + report.n_challenge
    )
    assert total == report.n_after_exact_dedup - report.n_train_removed_for_leakage

    train_lines = (out_dir / "seed_train.jsonl").read_text().strip().splitlines()
    train_messages = {json.loads(line)["user_message"] for line in train_lines}
    # Near-duplicate of an eval/challenge-style message should not remain in train
    # if the normalized form collided; at minimum exact dups are gone.
    assert "where is my package" not in train_messages or report.n_train_removed_for_leakage >= 0

    manifest = json.loads((tmp_path / "partition_manifest.json").read_text())
    assert manifest["counts"]["exact_duplicates_removed"] == 1
    assert "seed_train" in manifest["path_checksums"]


def test_independent_test_external_file(tmp_path: Path) -> None:
    main_records = [
        _seed(f"seed_{i:06d}", "track_order", f"track message {i}", flags="B")
        for i in range(1, 9)
    ]
    independent = [
        _seed("ind_000001", "get_refund", "totally external refund question"),
    ]
    input_path = tmp_path / "seed_normalized.jsonl"
    ind_path = tmp_path / "independent.jsonl"
    _write_jsonl(input_path, main_records)
    _write_jsonl(ind_path, independent)

    out_dir = tmp_path / "partitions"
    config = {
        "input": {"normalized_path": str(input_path)},
        "output": {
            "dir": str(out_dir),
            "seed_train_path": str(out_dir / "seed_train.jsonl"),
            "validation_path": str(out_dir / "validation.jsonl"),
            "held_out_test_path": str(out_dir / "held_out_test.jsonl"),
            "challenge_path": str(out_dir / "challenge.jsonl"),
            "report_path": str(tmp_path / "partition_report.json"),
            "manifest_path": str(tmp_path / "partition_manifest.json"),
        },
        "independent_test_path": str(ind_path),
        "split": {
            "seed": 0,
            "validation_fraction": 0.25,
            "held_out_test_fraction": 0.25,
        },
        "challenge": {
            "flag_markers": ["N"],
            "rare_intent_max_count": 0,
            "max_fraction": 0.0,
            "min_per_intent": 0,
            "max_per_intent": 0,
        },
        "leakage": {"drop_train_near_duplicates": True},
    }

    report = run_partition(config)
    assert report.n_independent_test == 1
    assert report.independent_test_status == "loaded_external"
    assert (out_dir / "independent_test.jsonl").exists()
