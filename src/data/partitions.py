"""Split normalized seeds into train/val/held-out/challenge sets (PRD 8.2)."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from data.ingest import checksum_file, write_json, write_jsonl
from data.schema import PartitionReport, SeedRecord

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/data/partitions.yaml")

PROVISIONAL_TEST_NOTE = (
    "held_out_test is a frozen Bitext hold-out for early experiments only. "
    "It is not an independent real-world test set (PRD 8.1). Replace or "
    "supplement with teacher-unavailable examples via independent_test_path."
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)

# load previously data pipeline downloaded and normalized seed records from local jsonl file.
def load_seed_records(path: Path) -> list[SeedRecord]:
    if not path.exists():
        raise FileNotFoundError(
            f"Normalized seed file not found: {path}. Run ingest-data first."
        )
    records: list[SeedRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(SeedRecord.model_validate_json(line)) # parse and validate each json line, since it suppose to have same schema as SeedRecord.
            except Exception as exc:  # noqa: BLE001 - report row then fail
                raise ValueError(f"Invalid SeedRecord on line {line_number}") from exc
    return records

# normalize msg from 'Hello, how are you?' to 'hello how are you', to help detect exact duplicates.
def normalize_message(text: str) -> str:
    """Deterministic fingerprint for exact/near-duplicate leakage checks."""
    lowered = text.lower().strip()
    no_punct = re.sub(r"[^\w\s]", "", lowered, flags=re.UNICODE)
    return re.sub(r"\s+", " ", no_punct).strip()


def dedupe_exact_user_messages(records: list[SeedRecord]) -> tuple[list[SeedRecord], int]:
    seen: set[str] = set()
    kept: list[SeedRecord] = []
    removed = 0
    for record in records:
        key = record.user_message.strip()
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept.append(record)
    return kept, removed


def _intent_counts(records: list[SeedRecord]) -> dict[str, int]:
    return dict(Counter(record.intent for record in records))


# based on raw data tag, determine which records are challenge cases: N=noise, Q=colloquial, W=offensive
def select_challenge_records(
    records: list[SeedRecord],
    *,
    flag_markers: list[str],
    rare_intent_max_count: int,
    max_fraction: float,
    min_per_intent: int,
    max_per_intent: int,
    rng: random.Random,
) -> tuple[list[SeedRecord], list[SeedRecord], dict[str, Any]]:
    """Carve challenge cases; return (challenge, remainder, selection_meta)."""
    counts = _intent_counts(records)
    rare_intents = {
        intent for intent, count in counts.items() if count < rare_intent_max_count
    }

    candidates: list[SeedRecord] = []
    remainder: list[SeedRecord] = []
    reasons = {"flag_marker": 0, "rare_intent": 0, "both": 0}

    for record in records:
        flag_hit = any(marker in (record.flags or "") for marker in flag_markers)
        rare_hit = record.intent in rare_intents
        if flag_hit or rare_hit:
            candidates.append(record)
            if flag_hit and rare_hit:
                reasons["both"] += 1
            elif flag_hit:
                reasons["flag_marker"] += 1
            else:
                reasons["rare_intent"] += 1
        else:
            remainder.append(record)

    max_challenge = max(0, int(len(records) * max_fraction)) # at most 8% of the records are challenge cases, so if there are 1000 records, then at most 80 are challenge cases.
    by_intent: dict[str, list[SeedRecord]] = defaultdict(list)
    for record in candidates:
        by_intent[record.intent].append(record)

    selected: list[SeedRecord] = []
    not_selected: list[SeedRecord] = []
    for intent, items in sorted(by_intent.items()):
        rng.shuffle(items)
        take = min(len(items), max_per_intent) # at most 80 records per intent are selected for challenge cases.
        if min_per_intent:
            take = max(take, min(min_per_intent, len(items)))
        selected.extend(items[:take])
        not_selected.extend(items[take:])

    rng.shuffle(selected)
    if len(selected) > max_challenge:
        not_selected.extend(selected[max_challenge:])
        selected = selected[:max_challenge]

    remainder.extend(not_selected)

    meta = {
        "flag_markers": flag_markers,
        "rare_intent_max_count": rare_intent_max_count,
        "rare_intents": sorted(rare_intents),
        "n_candidates": len(candidates),
        "n_selected": len(selected),
        "max_challenge": max_challenge,
        "selection_reasons": reasons,
    }
    return selected, remainder, meta

# split records into train/val/held-out sets 80/10/10%, stratified by intent.
def stratified_three_way_split(
    records: list[SeedRecord],
    *,
    validation_fraction: float,
    held_out_test_fraction: float,
    rng: random.Random,
) -> tuple[list[SeedRecord], list[SeedRecord], list[SeedRecord]]:
    """Split by intent into seed_train / validation / held_out_test."""
    if validation_fraction < 0 or held_out_test_fraction < 0:
        raise ValueError("Split fractions must be non-negative")
    if validation_fraction + held_out_test_fraction >= 1.0:
        raise ValueError("validation_fraction + held_out_test_fraction must be < 1")

    by_intent: dict[str, list[SeedRecord]] = defaultdict(list)
    for record in records:
        by_intent[record.intent].append(record)

    train: list[SeedRecord] = []
    validation: list[SeedRecord] = []
    held_out: list[SeedRecord] = []

    for intent in sorted(by_intent):
        items = list(by_intent[intent])
        rng.shuffle(items)
        n = len(items)
        n_val = int(n * validation_fraction)
        n_test = int(n * held_out_test_fraction)
        # Keep at least one training example when the intent has enough rows.
        if n >= 3 and n_val + n_test >= n:
            n_test = max(0, n - n_val - 1)
        validation.extend(items[:n_val])
        held_out.extend(items[n_val : n_val + n_test])
        train.extend(items[n_val + n_test :])

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(held_out)
    return train, validation, held_out


def drop_train_leakage(
    train: list[SeedRecord],
    protected: list[SeedRecord],
) -> tuple[list[SeedRecord], int]:
    """Remove train rows whose normalized message collides with eval partitions."""
    protected_keys = {normalize_message(record.user_message) for record in protected}
    protected_keys.discard("")
    cleaned: list[SeedRecord] = []
    removed = 0
    for record in train:
        key = normalize_message(record.user_message)
        if key in protected_keys:
            removed += 1
            continue
        cleaned.append(record)
    return cleaned, removed


def _counts_by_split(splits: dict[str, list[SeedRecord]]) -> dict[str, dict[str, int]]:
    return {name: _intent_counts(rows) for name, rows in splits.items()}


def run_partition(config: dict[str, Any]) -> PartitionReport:
    created_at = datetime.now(timezone.utc)
    input_path = Path(config["input"]["normalized_path"])
    output = config["output"]
    split_cfg = config["split"]
    challenge_cfg = config["challenge"]
    leakage_cfg = config.get("leakage", {})
    rng = random.Random(int(split_cfg.get("seed", 42)))

    records = load_seed_records(input_path)
    n_input = len(records)
    records, n_dup_removed = dedupe_exact_user_messages(records) # remove exact duplicates from the records

    challenge, remainder, challenge_meta = select_challenge_records(
        records,
        flag_markers=list(challenge_cfg.get("flag_markers", [])),
        rare_intent_max_count=int(challenge_cfg.get("rare_intent_max_count", 800)),
        max_fraction=float(challenge_cfg.get("max_fraction", 0.08)),
        min_per_intent=int(challenge_cfg.get("min_per_intent", 0)),
        max_per_intent=int(challenge_cfg.get("max_per_intent", 80)),
        rng=rng,
    )

    seed_train, validation, held_out_test = stratified_three_way_split(
        remainder,
        validation_fraction=float(split_cfg["validation_fraction"]),
        held_out_test_fraction=float(split_cfg["held_out_test_fraction"]),
        rng=rng,
    )

    n_leakage_removed = 0
    # remove train rows whose normalized message collides with eval partitions.
    if leakage_cfg.get("drop_train_near_duplicates", True):
        seed_train, n_leakage_removed = drop_train_leakage(
            seed_train,
            protected=validation + held_out_test + challenge,
        )

    independent_records: list[SeedRecord] = []
    independent_status = "pending_external_source"
    independent_path = config.get("independent_test_path")
    if independent_path:
        independent_records = load_seed_records(Path(independent_path))
        independent_status = "loaded_external"
        if leakage_cfg.get("drop_train_near_duplicates", True):
            seed_train, extra_removed = drop_train_leakage(
                seed_train,
                protected=independent_records,
            )
            n_leakage_removed += extra_removed

    Path(output["dir"]).mkdir(parents=True, exist_ok=True)
    paths = {
        "seed_train": Path(output["seed_train_path"]),
        "validation": Path(output["validation_path"]),
        "held_out_test": Path(output["held_out_test_path"]),
        "challenge": Path(output["challenge_path"]),
    }
    write_jsonl(seed_train, paths["seed_train"])
    write_jsonl(validation, paths["validation"])
    write_jsonl(held_out_test, paths["held_out_test"])
    write_jsonl(challenge, paths["challenge"])

    independent_out: Path | None = None
    if independent_records:
        independent_out = Path(output["dir"]) / "independent_test.jsonl"
        write_jsonl(independent_records, independent_out)

    splits = {
        "seed_train": seed_train,
        "validation": validation,
        "held_out_test": held_out_test,
        "challenge": challenge,
    }
    if independent_records:
        splits["independent_test"] = independent_records

    notes = [PROVISIONAL_TEST_NOTE]
    if independent_status == "pending_external_source":
        notes.append(
            "No independent_test_path configured; generation teacher must not see "
            "held_out_test or challenge prompts."
        )

    report = PartitionReport(
        created_at=created_at,
        input_path=str(input_path),
        n_input_rows=n_input,
        n_after_exact_dedup=len(records),
        n_exact_duplicates_removed=n_dup_removed,
        n_seed_train=len(seed_train),
        n_validation=len(validation),
        n_held_out_test=len(held_out_test),
        n_challenge=len(challenge),
        n_independent_test=len(independent_records),
        n_train_removed_for_leakage=n_leakage_removed,
        intent_counts_by_split=_counts_by_split(splits),
        challenge_selection=challenge_meta,
        independent_test_status=independent_status,
        notes=notes,
        extras={
            "split_seed": split_cfg.get("seed", 42),
            "validation_fraction": split_cfg.get("validation_fraction"),
            "held_out_test_fraction": split_cfg.get("held_out_test_fraction"),
        },
    )

    report_path = Path(output["report_path"])
    manifest_path = Path(output["manifest_path"])
    write_json(report.model_dump(mode="json"), report_path)

    manifest = {
        "created_at": created_at.isoformat(),
        "input_path": str(input_path),
        "input_sha256": checksum_file(input_path),
        "split_seed": split_cfg.get("seed", 42),
        "counts": {
            "seed_train": len(seed_train),
            "validation": len(validation),
            "held_out_test": len(held_out_test),
            "challenge": len(challenge),
            "independent_test": len(independent_records),
            "exact_duplicates_removed": n_dup_removed,
            "train_removed_for_leakage": n_leakage_removed,
        },
        "paths": {name: str(path) for name, path in paths.items()},
        "path_checksums": {
            name: checksum_file(path) for name, path in paths.items()
        },
        "independent_test_status": independent_status,
        "independent_test_path": str(independent_out) if independent_out else None,
        "notes": notes,
    }
    write_json(manifest, manifest_path)

    logger.info(
        "Partitioned %s rows → train=%s val=%s held_out=%s challenge=%s "
        "(dedup_removed=%s leakage_removed=%s)",
        n_input,
        len(seed_train),
        len(validation),
        len(held_out_test),
        len(challenge),
        n_dup_removed,
        n_leakage_removed,
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Partition normalized seeds into train/val/held-out/challenge sets (PRD 8.2)."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to partitions YAML config",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional override for normalized seed JSONL",
    )
    parser.add_argument(
        "--independent-test",
        type=str,
        default=None,
        help="Optional external independent-test JSONL",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)

    if args.input:
        config.setdefault("input", {})["normalized_path"] = args.input
    if args.independent_test:
        config["independent_test_path"] = args.independent_test

    report = run_partition(config)
    print(
        json.dumps(
            {
                "n_seed_train": report.n_seed_train,
                "n_validation": report.n_validation,
                "n_held_out_test": report.n_held_out_test,
                "n_challenge": report.n_challenge,
                "n_independent_test": report.n_independent_test,
                "independent_test_status": report.independent_test_status,
                "report_path": config["output"]["report_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
