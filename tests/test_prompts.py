"""Offline tests for versioned generation prompts (PRD 8.4)."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone

from generation.prompts import (
    PROMPT_VERSION,
    VARIATION_TYPES,
    build_chat_messages,
    plan_generation_jobs,
    sample_variation_types,
)
from data.schema import SeedRecord


def _seed() -> SeedRecord:
    return SeedRecord(
        seed_id="seed_000001",
        category="REFUND",
        intent="get_refund",
        user_message="I want a refund for my broken item",
        assistant_response="I can help with a refund request.",
        flags="B",
        source_dataset="test",
        retrieved_at=datetime.now(timezone.utc),
    )


def test_build_chat_messages_contains_seed_and_policy() -> None:
    messages = build_chat_messages(_seed(), "paraphrase")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Support policy" in messages[0]["content"]
    assert "JSON object" in messages[0]["content"]
    user = json.loads(messages[1]["content"])
    assert user["variation_type"] == "paraphrase"
    assert user["seed"]["seed_id"] == "seed_000001"
    assert user["seed"]["intent"] == "get_refund"


def test_unknown_variation_raises() -> None:
    try:
        build_chat_messages(_seed(), "not_a_type")
    except ValueError as exc:
        assert "Unknown variation_type" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_plan_generation_jobs_uses_only_given_seeds() -> None:
    rng = random.Random(42)
    weights = {name: 1.0 for name in VARIATION_TYPES}
    jobs = plan_generation_jobs([_seed()], n_examples=5, weights=weights, rng=rng)
    assert len(jobs) == 5
    assert jobs[0]["prompt_version"] == PROMPT_VERSION
    assert all(job["parent_seed_id"] == "seed_000001" for job in jobs)
    assert {job["variation_type"] for job in jobs} <= set(VARIATION_TYPES)


def test_sample_variation_types_respects_zero_weight() -> None:
    rng = random.Random(0)
    weights = {"paraphrase": 1.0, "safety_adversarial": 0.0}
    drawn = sample_variation_types(20, weights, rng)
    assert set(drawn) == {"paraphrase"}
