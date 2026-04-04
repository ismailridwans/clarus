"""SQLite schema creation for Clarus environment.

Two categories of tables:
- Runtime tables: wiped on reset(), one per episode.
- Reference tables: loaded once at startup from CMS data, never wiped.
"""

import sqlite3


def create_tables(db: sqlite3.Connection) -> None:
    """Create all runtime and reference tables.

    Idempotent — safe to call multiple times on the same connection.
    Uses CREATE TABLE IF NOT EXISTS for reference tables and creates
    runtime tables fresh each time via DROP + CREATE during reset().
    """
    db.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=ON;

        -- ----------------------------------------------------------------
        -- Runtime tables (per-episode, wiped at reset)
        -- ----------------------------------------------------------------

        CREATE TABLE IF NOT EXISTS episode_artifacts (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id    TEXT    NOT NULL,
            artifact_type TEXT    NOT NULL,
            source        TEXT    NOT NULL,
            content       TEXT    NOT NULL,
            created_at    INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS action_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id  TEXT    NOT NULL,
            step_number INTEGER NOT NULL,
            action_type TEXT    NOT NULL,
            parameters  TEXT    NOT NULL,
            result      TEXT,
            error       TEXT
        );

        CREATE TABLE IF NOT EXISTS compliance_events (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            episode_id     TEXT    NOT NULL,
            step_number    INTEGER NOT NULL,
            violation_type TEXT    NOT NULL,
            description    TEXT    NOT NULL
        );

        -- ----------------------------------------------------------------
        -- Reference tables (real CMS data, loaded once, never wiped)
        -- ----------------------------------------------------------------

        CREATE TABLE IF NOT EXISTS cpt_codes (
            code        TEXT PRIMARY KEY,
            short_desc  TEXT NOT NULL,
            long_desc   TEXT,
            rvu_work    REAL,
            rvu_total   REAL,
            status      TEXT
        );

        CREATE TABLE IF NOT EXISTS ncci_edits (
            col1_code          TEXT    NOT NULL,
            col2_code          TEXT    NOT NULL,
            effective_date     TEXT    NOT NULL,
            deletion_date      TEXT,
            modifier_indicator INTEGER NOT NULL,
            PRIMARY KEY (col1_code, col2_code)
        );

        CREATE TABLE IF NOT EXISTS carc_codes (
            code        TEXT PRIMARY KEY,
            category    TEXT NOT NULL,
            description TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS nsa_qpa_rates (
            procedure_code  TEXT    NOT NULL,
            geographic_area TEXT    NOT NULL,
            median_rate     REAL    NOT NULL,
            year            INTEGER NOT NULL,
            source          TEXT    NOT NULL,
            PRIMARY KEY (procedure_code, geographic_area, year)
        );

        CREATE TABLE IF NOT EXISTS plan_templates (
            plan_id               TEXT PRIMARY KEY,
            plan_type             TEXT NOT NULL,
            deductible_individual REAL NOT NULL,
            coinsurance_rate      REAL NOT NULL,
            copay_specialist      REAL NOT NULL,
            oop_max               REAL NOT NULL,
            nsa_compliant         INTEGER NOT NULL
        );

        -- ----------------------------------------------------------------
        -- Indexes
        -- ----------------------------------------------------------------

        CREATE INDEX IF NOT EXISTS idx_art_ep_type
            ON episode_artifacts(episode_id, artifact_type);

        CREATE INDEX IF NOT EXISTS idx_log_ep
            ON action_log(episode_id);

        CREATE INDEX IF NOT EXISTS idx_ncci_col1
            ON ncci_edits(col1_code);

        CREATE INDEX IF NOT EXISTS idx_qpa_proc
            ON nsa_qpa_rates(procedure_code);
        """
    )
    db.commit()


def wipe_episode_data(db: sqlite3.Connection, episode_id: str) -> None:
    """Delete all runtime rows for a specific episode.

    Called at reset() to ensure each episode starts with a clean slate
    while preserving reference data.
    """
    db.execute(
        "DELETE FROM episode_artifacts WHERE episode_id = ?", (episode_id,)
    )
    db.execute("DELETE FROM action_log WHERE episode_id = ?", (episode_id,))
    db.execute(
        "DELETE FROM compliance_events WHERE episode_id = ?", (episode_id,)
    )
    db.commit()
