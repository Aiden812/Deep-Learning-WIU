"""Estimate out-of-pocket treatment cost after Singapore subsidies (e.g. CHAS).

Inputs:  condition name(s) from shared/conditions.yaml + patient subsidy tier
Data:    cost/procedure_mapping.csv, cost/subsidy_tables/*.csv (each row sourced + dated)
"""

from pathlib import Path

COST_DIR = Path(__file__).resolve().parent
PROCEDURE_MAPPING = COST_DIR / "procedure_mapping.csv"
SUBSIDY_TABLES = COST_DIR / "subsidy_tables"


def estimate(conditions: list[str], subsidy_tier: str | None = None) -> dict:
    """Return {"procedures": [...], "gross_min", "gross_max", "subsidy", "net_min", "net_max", "sources": [...]}."""
    raise NotImplementedError
