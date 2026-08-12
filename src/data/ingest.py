"""for synthetic data generation, download or load the Bitext seed dataset, validate, normalize, and report (FR-1)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import ValidationError

from data.schema import (
    BITEXT_INTENTS,
    BITEXT_REQUIRED_COLUMNS,
    QualityReport,
    SeedRecord,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/data/ingest.yaml")

TRANSFORMATION_STEPS = [
    "Load raw table from Hugging Face Hub or a local file",
    "Validate presence of Bitext required columns",
    "Drop rows with empty category, intent, instruction, or response",
    "Map instruction→user_message and response→assistant_response",
    "Assign stable seed_id values (seed_000001, …)",
    "Attach source dataset id, revision, and retrieval timestamp",
    "Write normalized JSONL, quality report, and manifest",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_raw_dataframe(
    *,
    local_path: str | None,
    hf_dataset_id: str,
    hf_split: str,
    hf_revision: str | None,
) -> tuple[pd.DataFrame, str, str | None]:
    """Return (dataframe, source_label, revision)."""
    if local_path:
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Local dataset not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix == ".parquet":
            frame = pd.read_parquet(path)
        elif suffix in {".json", ".jsonl"}:
            frame = pd.read_json(path, lines=suffix == ".jsonl")
        else:
            raise ValueError(f"Unsupported local file type: {suffix}")
        return frame, str(path.resolve()), None

    from datasets import load_dataset

    dataset = load_dataset(
        hf_dataset_id,
        split=hf_split,
        revision=hf_revision,
    )
    revision = hf_revision or getattr(dataset, "_fingerprint", None)
    frame = dataset.to_pandas()
    return frame, hf_dataset_id, revision if isinstance(revision, str) else hf_revision


def missing_required_columns(columns: list[str]) -> list[str]:
    present = {column.lower() for column in columns}
    # Bitext columns are lowercase; tolerate case differences.
    return sorted(col for col in BITEXT_REQUIRED_COLUMNS if col not in present)

# in case the column names are not in the correct format such as "   Instruction  "
def _normalize_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {column: column.strip().lower() for column in frame.columns}
    return frame.rename(columns=rename)


def validate_raw_schema(frame: pd.DataFrame) -> list[str]:
    missing = missing_required_columns(list(frame.columns))
    if missing:
        raise ValueError(
            "Seed dataset is missing required Bitext columns: " + ", ".join(missing)
        )
    return missing


def _is_blank(value: Any) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    return str(value).strip() == ""

# just rename the columns and use SeedRecord to validate the data for each record is all valid.
def normalize_records(
    frame: pd.DataFrame,
    *,
    source_dataset: str,
    source_revision: str | None,
    retrieved_at: datetime,
) -> tuple[list[SeedRecord], int]:
    """Normalize valid rows; return (records, n_dropped)."""
    records: list[SeedRecord] = []
    dropped = 0

    for index, row in frame.iterrows():
        raw = {
            "category": row.get("category"),
            "intent": row.get("intent"),
            "user_message": row.get("instruction"),
            "assistant_response": row.get("response"),
            "flags": "" if _is_blank(row.get("flags")) else str(row.get("flags")),
            "source_dataset": source_dataset,
            "source_revision": source_revision,
            "retrieved_at": retrieved_at,
            "seed_id": f"seed_{len(records) + 1:06d}",
        }
        try:
            records.append(SeedRecord.model_validate(raw))
        except ValidationError:
            dropped += 1
            logger.debug("Dropped invalid row at index %s", index)

    return records, dropped


def _length_stats(values: list[str]) -> dict[str, float]:
    lengths = [len(value) for value in values]
    if not lengths:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0}
    series = pd.Series(lengths)
    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "p50": float(series.median()),
    }

# try to build the report related to data source, drop, duplicates, intents num and categories num, etc.
def build_quality_report(
    raw: pd.DataFrame,
    records: list[SeedRecord],
    *,
    source_dataset: str,
    source_revision: str | None,
    retrieved_at: datetime,
    n_dropped: int,
    licence_note: str,
    provenance_note: str,
) -> QualityReport:
    null_or_empty: dict[str, int] = {}
    for column in BITEXT_REQUIRED_COLUMNS:
        if column not in raw.columns:
            null_or_empty[column] = len(raw)
            continue
        null_or_empty[column] = int(raw[column].map(_is_blank).sum())

    user_messages = [record.user_message for record in records]
    pairs = [(record.user_message, record.assistant_response) for record in records]
    intent_counts = (
        pd.Series([record.intent for record in records]).value_counts().to_dict()
        if records
        else {}
    )
    category_counts = (
        pd.Series([record.category for record in records]).value_counts().to_dict()
        if records
        else {}
    )
    unknown = sorted(
        intent for intent in intent_counts if intent not in BITEXT_INTENTS
    )

    return QualityReport(
        source_dataset=source_dataset,
        source_revision=source_revision,
        retrieved_at=retrieved_at,
        licence_note=licence_note.strip(),
        provenance_note=provenance_note.strip(),
        n_rows_raw=len(raw),
        n_rows_normalized=len(records),
        n_rows_dropped=n_dropped,
        columns_raw=list(raw.columns),
        missing_required_columns=[],
        null_or_empty_counts=null_or_empty,
        n_exact_duplicate_user_messages=int(
            len(user_messages) - len(set(user_messages))
        ),
        n_exact_duplicate_pairs=int(len(pairs) - len(set(pairs))),
        intent_counts={str(k): int(v) for k, v in intent_counts.items()},
        category_counts={str(k): int(v) for k, v in category_counts.items()},
        unknown_intents=unknown,
        user_message_length=_length_stats(user_messages),
        assistant_response_length=_length_stats(
            [record.assistant_response for record in records]
        ),
        transformation_steps=list(TRANSFORMATION_STEPS),
        extras={
            "n_intents": len(intent_counts),
            "n_categories": len(category_counts),
            "documented_intent_coverage": round(
                len([i for i in intent_counts if i in BITEXT_INTENTS])
                / max(len(BITEXT_INTENTS), 1),
                4,
            ),
        },
    )


def checksum_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

# keep normalized data in jsonl format in disk, so later train, test, eval commands can use it.
def write_jsonl(records: list[SeedRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")


def run_ingest(config: dict[str, Any]) -> QualityReport:
    source = config["source"]
    output = config["output"]
    retrieved_at = datetime.now(timezone.utc)

    raw, source_label, revision = load_raw_dataframe(
        local_path=source.get("local_path"),
        hf_dataset_id=source["hf_dataset_id"],
        hf_split=source.get("hf_split", "train"),
        hf_revision=source.get("hf_revision"),
    )
    raw = _normalize_column_names(raw)
    validate_raw_schema(raw)

    records, n_dropped = normalize_records(
        raw,
        source_dataset=source_label,
        source_revision=revision,
        retrieved_at=retrieved_at,
    )
    report = build_quality_report(
        raw,
        records,
        source_dataset=source_label,
        source_revision=revision,
        retrieved_at=retrieved_at,
        n_dropped=n_dropped,
        licence_note=config.get("licence_note", ""),
        provenance_note=config.get("provenance_note", ""),
    )

    normalized_path = Path(output["normalized_path"])
    report_path = Path(output["report_path"])
    manifest_path = Path(output["manifest_path"])

    write_jsonl(records, normalized_path)
    write_json(report.model_dump(mode="json"), report_path)

    manifest = {
        "created_at": retrieved_at.isoformat(),
        "source_dataset": source_label,
        "source_revision": revision,
        "n_rows": len(records),
        "n_intents": report.extras.get("n_intents"),
        "n_categories": report.extras.get("n_categories"),
        "normalized_path": str(normalized_path),
        "report_path": str(report_path),
        "normalized_sha256": checksum_file(normalized_path),
        "config_transformation_steps": TRANSFORMATION_STEPS,
    }
    write_json(manifest, manifest_path)

    logger.info(
        "Ingested %s rows → %s (%s dropped). Report: %s",
        len(records),
        normalized_path,
        n_dropped,
        report_path,
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest and profile the Bitext customer-support seed dataset (FR-1)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to ingest YAML config",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="Optional local CSV/JSON/JSONL/Parquet override",
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        default=None,
        help="Optional Hugging Face dataset revision override",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)

    if args.local_path:
        config.setdefault("source", {})["local_path"] = args.local_path
    if args.hf_revision:
        config.setdefault("source", {})["hf_revision"] = args.hf_revision

    report = run_ingest(config)
    print(
        json.dumps(
            {
                "n_rows_normalized": report.n_rows_normalized,
                "n_rows_dropped": report.n_rows_dropped,
                "n_intents": report.extras.get("n_intents"),
                "report_path": config["output"]["report_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
