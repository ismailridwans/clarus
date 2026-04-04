"""Print the key values generated for a set of seeds.

Run this before writing test assertions to verify generator output.
This script never asserts anything — it just prints so you can verify.

Usage:
    python scripts/discover_seed_values.py
    python scripts/discover_seed_values.py --seeds 1001,2001,3001
"""

from __future__ import annotations

import argparse
import sqlite3
import sys


def make_ref_db() -> sqlite3.Connection:
    """Build an in-memory reference DB with all bundle data."""
    from server.schema import create_tables
    from data.setup import load_all

    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.row_factory = sqlite3.Row
    create_tables(db)
    load_all(db)
    return db


def main(seeds: list[int] | None = None) -> None:
    """Print generator output for given seeds (or default discovery set)."""
    from server.scenario.generator import generate

    ref_db = make_ref_db()

    if seeds is None:
        seeds = [1001, 1002, 1003, 2001, 2002, 3001, 3002]

    task_map = {
        1001: "deductive_liability", 1002: "deductive_liability", 1003: "deductive_liability",
        2001: "abductive_conflict", 2002: "abductive_conflict",
        3001: "adversarial_fabrication", 3002: "adversarial_fabrication",
    }

    print(f"{'seed':>6}  {'task':<28}  {'cpt_primary':<12}  {'service_date':<12}  "
          f"{'billed':>8}  {'refund':>8}  {'qpa':>8}")
    print("-" * 100)

    for seed in seeds:
        # Auto-detect task from seed range
        if seed < 2000:
            task = "deductive_liability"
        elif seed < 3000:
            task = "abductive_conflict"
        else:
            task = "adversarial_fabrication"

        p = generate(task, seed, ref_db)
        print(
            f"{seed:>6}  {task:<28}  {p.cpt_primary:<12}  {p.service_date:<12}  "
            f"{p.billed_amount:>8.2f}  {p.correct_refund:>8.2f}  {p.qpa_amount:>8.2f}"
        )
        if task == "deductive_liability":
            print(
                f"         deductible={p.deductible:.0f}  met={p.deductible_met:.2f}  "
                f"coinsurance={p.coinsurance_rate:.0%}  copay={p.copay_specialist:.0f}  "
                f"correct_balance={p.correct_balance:.2f}  "
                f"has_deposit={p.has_scheduling_deposit}"
            )
        elif task == "abductive_conflict":
            print(
                f"         cpt_primary={p.cpt_primary}  cpt_secondary={p.cpt_secondary}  "
                f"modifier={p.modifier_used}  denial={p.denial_code}  "
                f"appeal_reason={p.appeal_reason_correct}"
            )
        elif task == "adversarial_fabrication":
            print(
                f"         gfe_fabricated={p.gfe_fabricated_date}  "
                f"processor_ts={p.processor_timestamp}  "
                f"qpa={p.qpa_amount:.2f}  billed_oon={p.billed_amount_oon:.2f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discover seed values")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds, e.g. 1001,2001,3001",
    )
    args = parser.parse_args()
    seeds = None
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
    main(seeds)
