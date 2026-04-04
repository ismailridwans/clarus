"""Load all reference data into a SQLite connection.

Parses the CSV/JSON files produced by download.py and inserts them
into the reference tables.  Called once at startup; also called inside
the Dockerfile self-check.

Usage:
    python data/setup.py          # validates local ref_db
"""

import csv
import json
import sqlite3
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent
BUNDLE_DIR = DATA_DIR / "bundles"


def _load_cpt_codes(db: sqlite3.Connection, csv_path: Path) -> int:
    """Parse CPT codes CSV and insert into cpt_codes table.

    Handles both the full CMS PPRRVU format and the bundle format.
    Returns number of rows inserted.
    """
    rows_inserted = 0
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []

        # Detect format: bundle has lowercase headers; CMS has mixed
        is_bundle = "code" in fieldnames

        batch = []
        for row in reader:
            if is_bundle:
                code = (row.get("code") or "").strip()
                short_desc = (row.get("short_desc") or "").strip()
                long_desc = (row.get("long_desc") or "").strip()
                try:
                    rvu_work = float(row.get("rvu_work") or 0)
                    rvu_total = float(row.get("rvu_total") or 0)
                except ValueError:
                    rvu_work = rvu_total = 0.0
                status = (row.get("status") or "A").strip()
            else:
                # CMS PPRRVU format — column names vary; use positional fallback
                code = (
                    row.get("HCPCS")
                    or row.get("CPT")
                    or row.get("code")
                    or ""
                ).strip()
                short_desc = (
                    row.get("DESCRIPTION")
                    or row.get("short_desc")
                    or ""
                ).strip()
                long_desc = ""
                try:
                    rvu_work = float(
                        row.get("WORK RVU") or row.get("rvu_work") or 0
                    )
                    rvu_total = float(
                        row.get("TOTAL RVU") or row.get("rvu_total") or 0
                    )
                except ValueError:
                    rvu_work = rvu_total = 0.0
                status = (row.get("STATUS") or row.get("status") or "A").strip()

            if not code:
                continue
            batch.append((code, short_desc, long_desc, rvu_work, rvu_total, status))

        db.executemany(
            "INSERT OR REPLACE INTO cpt_codes "
            "(code, short_desc, long_desc, rvu_work, rvu_total, status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            batch,
        )
        rows_inserted = len(batch)

    db.commit()
    return rows_inserted


def _load_ncci_edits(db: sqlite3.Connection, csv_path: Path) -> int:
    """Parse NCCI PtP edits CSV and insert into ncci_edits table.

    Handles both bundle format (lowercase) and CMS format (mixed case).
    Returns number of rows inserted.
    """
    rows_inserted = 0
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        is_bundle = "col1_code" in fieldnames

        batch = []
        for row in reader:
            if is_bundle:
                col1 = (row.get("col1_code") or "").strip()
                col2 = (row.get("col2_code") or "").strip()
                eff = (row.get("effective_date") or "").strip()
                del_date = (row.get("deletion_date") or "").strip() or None
                try:
                    mod_ind = int(row.get("modifier_indicator") or 0)
                except ValueError:
                    mod_ind = 0
            else:
                # CMS format — columns may be: Col 1, Col 2, Effective Date, etc.
                col1 = (
                    row.get("Col 1") or row.get("col1_code") or ""
                ).strip()
                col2 = (
                    row.get("Col 2") or row.get("col2_code") or ""
                ).strip()
                eff = (
                    row.get("Effective Date") or row.get("effective_date") or ""
                ).strip()
                del_date = (
                    row.get("Deletion Date") or row.get("deletion_date") or ""
                ).strip() or None
                try:
                    mod_ind = int(
                        row.get("Modifier Indicator")
                        or row.get("modifier_indicator")
                        or 0
                    )
                except ValueError:
                    mod_ind = 0

            if not col1 or not col2:
                continue
            batch.append((col1, col2, eff, del_date, mod_ind))

        db.executemany(
            "INSERT OR REPLACE INTO ncci_edits "
            "(col1_code, col2_code, effective_date, deletion_date, modifier_indicator) "
            "VALUES (?, ?, ?, ?, ?)",
            batch,
        )
        rows_inserted = len(batch)

    db.commit()
    return rows_inserted


def _load_nsa_qpa_rates(db: sqlite3.Connection, csv_path: Path) -> int:
    """Parse NSA QPA rates CSV and insert into nsa_qpa_rates table.

    Returns number of rows inserted.
    """
    rows_inserted = 0
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        batch = []
        for row in reader:
            proc = (row.get("procedure_code") or "").strip()
            geo = (row.get("geographic_area") or "").strip()
            try:
                rate = float(row.get("median_rate") or 0)
                year = int(row.get("year") or 2026)
            except ValueError:
                continue
            source = (row.get("source") or "CMS_MPFS_2026").strip()
            if not proc or not geo:
                continue
            batch.append((proc, geo, rate, year, source))

        db.executemany(
            "INSERT OR REPLACE INTO nsa_qpa_rates "
            "(procedure_code, geographic_area, median_rate, year, source) "
            "VALUES (?, ?, ?, ?, ?)",
            batch,
        )
        rows_inserted = len(batch)

    db.commit()
    return rows_inserted


def _load_carc_codes(db: sqlite3.Connection) -> int:
    """Load CARC codes from committed JSON bundle.

    Always uses the bundle (no CMS download for this file).
    Returns number of rows inserted.
    """
    carc_path = BUNDLE_DIR / "carc_codes.json"
    with open(carc_path, encoding="utf-8") as fh:
        codes = json.load(fh)

    batch = [
        (c["code"], c["category"], c["description"]) for c in codes
    ]
    db.executemany(
        "INSERT OR REPLACE INTO carc_codes (code, category, description) "
        "VALUES (?, ?, ?)",
        batch,
    )
    db.commit()
    return len(batch)


def _load_plan_templates(db: sqlite3.Connection) -> int:
    """Load plan templates from committed JSON bundle.

    Returns number of rows inserted.
    """
    plan_path = BUNDLE_DIR / "plan_templates.json"
    with open(plan_path, encoding="utf-8") as fh:
        plans = json.load(fh)

    batch = [
        (
            p["plan_id"],
            p["plan_type"],
            float(p["deductible_individual"]),
            float(p["coinsurance_rate"]),
            float(p["copay_specialist"]),
            float(p["oop_max"]),
            int(p["nsa_compliant"]),
        )
        for p in plans
    ]
    db.executemany(
        "INSERT OR REPLACE INTO plan_templates "
        "(plan_id, plan_type, deductible_individual, coinsurance_rate, "
        " copay_specialist, oop_max, nsa_compliant) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        batch,
    )
    db.commit()
    return len(batch)


def load_all(db: sqlite3.Connection) -> dict:
    """Load all reference data into db.  Returns row counts per table.

    Expects download.py to have already placed data files in DATA_DIR.
    """
    from server.schema import create_tables  # noqa: local import avoids circular

    create_tables(db)

    cpt_path = DATA_DIR / "cpt_codes.csv"
    ncci_path = DATA_DIR / "ncci_edits.csv"
    nsa_path = DATA_DIR / "nsa_qpa_rates.csv"

    # Fall back to bundles if download step was skipped
    if not cpt_path.exists():
        import shutil
        shutil.copy(BUNDLE_DIR / "cpt_codes_bundle.csv", cpt_path)
    if not ncci_path.exists():
        import shutil
        shutil.copy(BUNDLE_DIR / "ncci_edits_bundle.csv", ncci_path)
    if not nsa_path.exists():
        import shutil
        shutil.copy(BUNDLE_DIR / "nsa_qpa_rates_bundle.csv", nsa_path)

    counts = {
        "cpt_codes": _load_cpt_codes(db, cpt_path),
        "ncci_edits": _load_ncci_edits(db, ncci_path),
        "nsa_qpa_rates": _load_nsa_qpa_rates(db, nsa_path),
        "carc_codes": _load_carc_codes(db),
        "plan_templates": _load_plan_templates(db),
    }
    return counts


if __name__ == "__main__":
    import sqlite3 as _sqlite3

    _db = _sqlite3.connect(":memory:")
    counts = load_all(_db)
    for table, n in counts.items():
        print(f"  {table}: {n} rows")

    # Assertions
    assert counts["cpt_codes"] > 0, "No CPT codes loaded"
    assert counts["ncci_edits"] > 0, "No NCCI edits loaded"
    assert counts["nsa_qpa_rates"] > 0, "No NSA QPA rates loaded"
    assert counts["carc_codes"] > 0, "No CARC codes loaded"
    assert counts["plan_templates"] == 8, "Expected 8 plan templates"

    print("data/setup.py self-check passed.", flush=True)
