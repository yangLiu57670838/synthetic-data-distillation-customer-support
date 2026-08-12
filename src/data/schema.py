# Pydantic models for seed and synthetic records used across the data pipeline.
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator


# Columns expected in the Bitext customer-support seed dataset.
BITEXT_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {"flags", "instruction", "category", "intent", "response"}
)

# Documented Bitext intents (27). Used for quality reporting, not hard rejection.
BITEXT_INTENTS: frozenset[str] = frozenset(
    {
        "cancel_order",
        "change_order",
        "change_shipping_address",
        "check_cancellation_fee",
        "check_invoice",
        "check_payment_methods",
        "check_refund_policy",
        "complaint",
        "contact_customer_service",
        "contact_human_agent",
        "create_account",
        "delete_account",
        "delivery_options",
        "delivery_period",
        "edit_account",
        "get_invoice",
        "get_refund",
        "newsletter_subscription",
        "payment_issue",
        "place_order",
        "recover_password",
        "registration_problems",
        "review",
        "set_up_shipping_address",
        "switch_account",
        "track_order",
        "track_refund",
    }
)

# the schema for the seed record, validate, trim, throw error for empty strings.
class SeedRecord(BaseModel):
    """Normalized seed example used for taxonomy, generation, and baseline SFT."""

    seed_id: str
    category: str
    intent: str
    user_message: str
    assistant_response: str
    flags: str = ""
    source_dataset: str
    source_revision: str | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("category", "intent", "user_message", "assistant_response")
    @classmethod
    def non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field must be a non-empty string")
        return cleaned


class QualityReport(BaseModel):
    """Data-quality report produced by FR-1 ingestion."""

    source_dataset: str
    source_revision: str | None
    retrieved_at: datetime
    licence_note: str
    provenance_note: str
    n_rows_raw: int
    n_rows_normalized: int
    n_rows_dropped: int
    columns_raw: list[str]
    missing_required_columns: list[str]
    null_or_empty_counts: dict[str, int]
    n_exact_duplicate_user_messages: int
    n_exact_duplicate_pairs: int
    intent_counts: dict[str, int]
    category_counts: dict[str, int]
    unknown_intents: list[str]
    user_message_length: dict[str, float]
    assistant_response_length: dict[str, float]
    transformation_steps: list[str]
    extras: dict[str, Any] = Field(default_factory=dict)
