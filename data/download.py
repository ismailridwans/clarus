"""Download CMS reference data or fall back to committed bundles.

Tries CMS URLs first.  If any download fails (network error, timeout,
bad status code) the entire CMS attempt is abandoned and the local
bundles committed to data/bundles/ are used instead.  Docker build
therefore succeeds offline.

Usage:
    python data/download.py
"""

import csv
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

BUNDLE_DIR = Path(__file__).parent / "bundles"
DATA_DIR = Path(__file__).parent

# Destination paths (written by this script)
CPT_CSV = DATA_DIR / "cpt_codes.csv"
NCCI_CSV = DATA_DIR / "ncci_edits.csv"
NSA_CSV = DATA_DIR / "nsa_qpa_rates.csv"

# CMS source URLs (best-effort; may 404 when CMS updates)
CMS_CPT_URL = (
    "https://www.cms.gov/files/zip/pprrvu26-jan.zip"
)
CMS_NCCI_URL = (
    "https://www.cms.gov/files/zip/ncci-ptp-edits-2026-q1.zip"
)

TIMEOUT = 30  # seconds


def _download_bytes(url: str) -> bytes:
    """Fetch URL and return raw bytes; raises on any error."""
    req = urllib.request.Request(url, headers={"User-Agent": "Clarus/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} for {url}")
        return resp.read()


def _try_cms_cpt() -> bool:
    """Attempt to download and extract CPT codes from CMS zip.

    Returns True on success, False on any failure.
    """
    try:
        print("  Trying CMS CPT download …", flush=True)
        data = _download_bytes(CMS_CPT_URL)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            # The zip contains a CSV with various names across years
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                return False
            content = zf.read(csv_names[0]).decode("utf-8", errors="replace")
        # Write raw; setup.py parses column positions
        CPT_CSV.write_text(content, encoding="utf-8")
        print(f"  CPT codes downloaded → {CPT_CSV}", flush=True)
        return True
    except Exception as exc:
        print(f"  CPT download failed: {exc}", flush=True)
        return False


def _try_cms_ncci() -> bool:
    """Attempt to download NCCI PtP edits from CMS zip.

    Returns True on success, False on any failure.
    """
    try:
        print("  Trying CMS NCCI download …", flush=True)
        data = _download_bytes(CMS_NCCI_URL)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                return False
            content = zf.read(csv_names[0]).decode("utf-8", errors="replace")
        NCCI_CSV.write_text(content, encoding="utf-8")
        print(f"  NCCI edits downloaded → {NCCI_CSV}", flush=True)
        return True
    except Exception as exc:
        print(f"  NCCI download failed: {exc}", flush=True)
        return False


def _use_bundles() -> None:
    """Copy committed bundle files to the data/ destination paths."""
    import shutil

    shutil.copy(BUNDLE_DIR / "cpt_codes_bundle.csv", CPT_CSV)
    shutil.copy(BUNDLE_DIR / "ncci_edits_bundle.csv", NCCI_CSV)
    shutil.copy(BUNDLE_DIR / "nsa_qpa_rates_bundle.csv", NSA_CSV)
    print("  Using committed bundle data.", flush=True)


def main() -> None:
    """Entry point: download CMS data or fall back to bundles."""
    print("=== Clarus data download ===", flush=True)

    cpt_ok = _try_cms_cpt()
    ncci_ok = _try_cms_ncci()

    if not cpt_ok or not ncci_ok:
        print("CMS download incomplete — using bundle fallback.", flush=True)
        _use_bundles()
    else:
        # NSA rates are always derived from bundles (CMS MPFS proxy formula)
        import shutil
        shutil.copy(BUNDLE_DIR / "nsa_qpa_rates_bundle.csv", NSA_CSV)
        print("CMS download complete.", flush=True)

    # Verify files exist and are non-empty
    for path in (CPT_CSV, NCCI_CSV, NSA_CSV):
        if not path.exists() or path.stat().st_size == 0:
            sys.exit(f"ERROR: {path} missing or empty after download step.")

    print("Data files ready.", flush=True)


if __name__ == "__main__":
    main()
