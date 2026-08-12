"""Seed-dataset ingestion, normalization, profiling, and partitioning."""

from data.ingest import main, run_ingest
from data.schema import QualityReport, SeedRecord

__all__ = ["SeedRecord", "QualityReport", "run_ingest", "main"]
