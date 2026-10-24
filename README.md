# Synthetic Support Distillation

Synthetic-data distillation pipeline for customer-support language models.

Requires **Python 3.11+**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
```

### Phase 1 (core + developer tools)

```bash
pip install -e ".[dev]"
```

### Optional extras

Install extras as you need them:

```bash
pip install -e ".[generation]"   # teacher API generation
pip install -e ".[filtering]"    # near-duplicate / leakage checks
pip install -e ".[training]"     # QLoRA / SFT (torch, transformers, peft, trl)
pip install -e ".[tracking]"     # MLflow and Weights & Biases
pip install -e ".[serving]"      # FastAPI and Gradio demo
```

Combine extras in one install:

```bash
pip install -e ".[dev,generation,filtering]"
```

Full runtime stack (without `dev`):

```bash
pip install -e ".[all]"
```

Everything for local development:

```bash
pip install -e ".[all,dev]"
```

> **Note:** `bitsandbytes` (in the `training` extra) is skipped on macOS. Use a Linux/CUDA environment for quantized QLoRA training.

## Ingest seed data (FR-1)

Downloads the Bitext customer-support dataset from Hugging Face (or a local file), validates the schema, normalizes fields, and writes a quality report.

```bash
# from Hugging Face (default config: configs/data/ingest.yaml)
ingest-data

# or from a local CSV / JSON / JSONL / Parquet file
ingest-data --local-path /path/to/bitext.csv

# optional: pin an immutable Hub revision
ingest-data --hf-revision <commit-sha>
```

Outputs:

- `data/processed/seed_normalized.jsonl`
- `reports/seed_quality_report.json`
- `reports/seed_manifest.json`

## Partition data (PRD 8.2)

After ingest, split the normalized seeds into train / validation / held-out test / challenge sets, with exact-dedup and leakage checks.

```bash
# uses configs/data/partitions.yaml (expects ingest output)
partition-data

# optional overrides
partition-data --input data/processed/seed_normalized.jsonl
partition-data --independent-test data/raw/independent_test.jsonl
```

Outputs:

- `data/processed/partitions/seed_train.jsonl`
- `data/processed/partitions/validation.jsonl`
- `data/processed/partitions/held_out_test.jsonl` (provisional Bitext hold-out)
- `data/processed/partitions/challenge.jsonl`
- `reports/partition_report.json`
- `reports/partition_manifest.json`

> Bitext `held_out_test` is **not** a true independent real-world test set. Provide `--independent-test` when you have teacher-unavailable examples.

## Generation prompts (FR-2 / PRD 8.4)

Prompt templates only — no teacher API call yet. Jobs are planned from `seed_train`.

```bash
python -c "from generation.prompts import PROMPT_VERSION; print(PROMPT_VERSION)"
```

Config: `configs/generation/generation.yaml` (`v1`, 200-example pilot, ~1000 main).

## Development commands

```bash
# tests
pytest
pytest --cov=src

# lint / format
ruff check .
ruff format .

# type check
mypy
```

## Project layout

See `PRD.md` for goals, experiments, and architecture. Source packages live under `src/`; configs under `configs/`.
