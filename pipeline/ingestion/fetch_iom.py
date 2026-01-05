"""
fetch_iom.py
================
Fetches IOM DTM displacement data from the public v3 API and stores RAW records in Postgres.

Key design points:
- Matches the working request convention:
    GET https://dtmapi.iom.int/v3/displacement/admin2?CountryName=Central%20African%20Republic
- API key is OPTIONAL. If IOM_PRIMARY_KEY exists, we include it; otherwise we omit it.
- Stores PCODES (admin1Pcode/admin2Pcode) to enable robust Admin+H3 mapping.
- Writes raw records (no premature aggregation) into: car_cewp.iom_dtm_raw
- Also writes a convenience CSV snapshot to data/processed/

Expected downstream:
- spatial_disaggregation.py will read iom_dtm_raw and do deterministic selection + disaggregation.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import text

# --- Import Centralized Utilities ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import PATHS, logger, load_configs, get_db_engine, upload_to_postgis  # noqa: E402

SCHEMA = "car_cewp"
TABLE_NAME = "iom_dtm_raw"
OUTPUT_FILENAME = "iom_displacement_data_raw.csv"


@dataclass(frozen=True)
class IOMConfig:
    base_url: str = "https://dtmapi.iom.int/v3"
    country_name: str = "Central African Republic"
    local_file: Optional[str] = None  # optional fallback CSV path relative to repo root


class IOMClientV3:
    """Thin HTTP client for IOM DTM v3."""

    def __init__(self, base_url: str, api_key: Optional[str] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        headers = {
            "Accept": "application/json",
            # Helps avoid some WAF rules that block default python UA
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        if api_key:
            headers["Ocp-Apim-Subscription-Key"] = api_key
        self.session.headers.update(headers)

    def get_admin2_displacement(self, country_name: str) -> list[dict[str, Any]]:
        url = f"{self.base_url}/displacement/admin2"
        params = {"CountryName": country_name}
        logger.info(f"IOM GET: {url} params={params}")

        resp = self.session.get(url, params=params, timeout=45)
        # If key is wrong, you can get 401/403 even though endpoint is public
        if resp.status_code in (401, 403):
            snippet = resp.text[:400] if resp.text else ""
            raise RuntimeError(
                f"IOM API blocked ({resp.status_code}). "
                f"If you set IOM_PRIMARY_KEY, verify it's valid. "
                f"Response snippet: {snippet}"
            )
        resp.raise_for_status()

        payload = resp.json()
        if isinstance(payload, dict) and isinstance(payload.get("result"), list):
            return payload["result"]
        if isinstance(payload, list):
            return payload
        raise RuntimeError(f"Unexpected IOM payload shape: {type(payload)}")

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass


def _load_iom_config(data_cfg: dict) -> IOMConfig:
    iom_cfg = (data_cfg or {}).get("iom_dtm", {}) or {}
    return IOMConfig(
        base_url=iom_cfg.get("base_url") or "https://dtmapi.iom.int/v3",
        country_name=iom_cfg.get("country_name") or "Central African Republic",
        local_file=iom_cfg.get("local_file"),
    )


def _standardize_admin2_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Standardize admin2 displacement records into a consistent RAW schema.
    We keep as much as we reasonably can, including PCODES and round metadata.
    """
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).copy()

    rename = {
        "admin0Name": "admin0_name",
        "admin0Pcode": "admin0_pcode",
        "admin1Name": "admin1_name",
        "admin1Pcode": "admin1_pcode",
        "admin2Name": "admin2_name",
        "admin2Pcode": "admin2_pcode",
        "numPresentIdpInd": "individuals",
        "reportingDate": "reporting_date",
        "yearReportingDate": "year_reporting",
        "monthReportingDate": "month_reporting",
        "roundNumber": "round_number",
        "displacementReason": "displacement_reason",
        "assessmentType": "assessment_type",
        "numberMales": "males",
        "numberFemales": "females",
        "operation": "operation",
    }
    df = df.rename(columns=rename)

    required_cols = [
        "id",
        "admin0_name",
        "admin0_pcode",
        "admin1_name",
        "admin1_pcode",
        "admin2_name",
        "admin2_pcode",
        "individuals",
        "reporting_date",
        "year_reporting",
        "month_reporting",
        "round_number",
        "displacement_reason",
        "assessment_type",
        "males",
        "females",
        "operation",
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = None

    # Parse dates
    df["reporting_date"] = pd.to_datetime(df["reporting_date"], errors="coerce").dt.date
    df = df.dropna(subset=["reporting_date"])

    # Coerce numeric (initial)
    for c in ["id", "individuals", "year_reporting", "month_reporting", "round_number", "males", "females"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Enforce integer semantics
    df["individuals"] = pd.to_numeric(df["individuals"], errors="coerce").fillna(0).astype("int64")
    df["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype("int64")

    # Tag source endpoint + granularity
    df["granularity"] = "admin2"
    df["source_endpoint"] = "/v3/displacement/admin2"

    keep = [
        "id",
        "reporting_date",
        "year_reporting",
        "month_reporting",
        "round_number",
        "operation",
        "displacement_reason",
        "assessment_type",
        "admin0_name",
        "admin0_pcode",
        "admin1_name",
        "admin1_pcode",
        "admin2_name",
        "admin2_pcode",
        "individuals",
        "males",
        "females",
        "granularity",
        "source_endpoint",
    ]
    return df[keep].copy()


def _coerce_integer_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce integer-like columns to avoid float serialization (e.g., '8.0').
    """
    nullable_int_cols = ["males", "females", "year_reporting", "month_reporting", "round_number"]
    for col in nullable_int_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            df[col] = s.astype(object).where(s.notna(), "")
    if "individuals" in df.columns:
        df["individuals"] = pd.to_numeric(df["individuals"], errors="coerce").fillna(0).astype("int64")
    return df


def _assert_no_float_strings(df: pd.DataFrame) -> None:
    """
    Fail fast if any integer columns contain float-looking strings like '8.0'.
    """
    check_cols = ["males", "females", "year_reporting", "month_reporting", "round_number", "individuals"]
    for col in check_cols:
        if col in df.columns:
            ser = df[col]
            bad_mask = ser.astype(str).str.match(r"^\d+\.0$")
            if bad_mask.any():
                samples = ser[bad_mask].astype(str).unique().tolist()
                raise ValueError(f"Column {col} contains float-like integers (e.g., {samples[:5]}). Fix coercion before upload.")


def _ensure_raw_table(engine) -> None:
    """Create schema/table if missing. If table exists but lacks 'id', drop & recreate."""
    logger.info(f"Ensuring schema/table exist: {SCHEMA}.{TABLE_NAME}")

    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))

        exists = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = :schema AND table_name = :table
            );
        """), {"schema": SCHEMA, "table": TABLE_NAME}).scalar()

        if exists:
            has_id = conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = :schema
                      AND table_name = :table
                      AND column_name = 'id'
                );
            """), {"schema": SCHEMA, "table": TABLE_NAME}).scalar()

            if not has_id:
                logger.warning(
                    f"{SCHEMA}.{TABLE_NAME} exists but has no 'id' column (old schema). "
                    "Dropping and recreating with new schema."
                )
                conn.execute(text(f'DROP TABLE IF EXISTS {SCHEMA}.{TABLE_NAME};'))

        conn.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {SCHEMA}.{TABLE_NAME} (
                    id BIGINT PRIMARY KEY,
                    reporting_date DATE NOT NULL,
                    year_reporting INTEGER,
                    month_reporting INTEGER,
                    round_number INTEGER,
                    operation TEXT,
                    displacement_reason TEXT,
                    assessment_type TEXT,
                    admin0_name TEXT,
                    admin0_pcode TEXT,
                    admin1_name TEXT,
                    admin1_pcode TEXT,
                    admin2_name TEXT,
                    admin2_pcode TEXT,
                    individuals BIGINT,
                    males BIGINT,
                    females BIGINT,
                    granularity TEXT,
                    source_endpoint TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
        )


def main() -> None:
    data_cfg, _, _ = load_configs()
    iom_cfg = _load_iom_config(data_cfg)

    env_path = PATHS["root"] / ".env"
    load_dotenv(env_path)
    api_key = os.getenv("IOM_PRIMARY_KEY")
    if api_key:
        logger.info("IOM_PRIMARY_KEY found; will include subscription header.")
    else:
        logger.info("IOM_PRIMARY_KEY not found; calling public endpoint without subscription header.")

    engine = get_db_engine()
    client = IOMClientV3(base_url=iom_cfg.base_url, api_key=api_key)

    try:
        logger.info(f"Fetching IOM DTM data for: {iom_cfg.country_name}")
        records = client.get_admin2_displacement(iom_cfg.country_name)
        df = _standardize_admin2_records(records)

        # Local fallback (only if API yields nothing)
        if df.empty and iom_cfg.local_file:
            local_path = PATHS["root"] / iom_cfg.local_file
            if local_path.exists():
                logger.warning(f"API empty. Using local file: {local_path}")
                df_local = pd.read_csv(local_path)
                if "reporting_date" in df_local.columns:
                    df_local["reporting_date"] = pd.to_datetime(df_local["reporting_date"], errors="coerce").dt.date
                if "id" not in df_local.columns:
                    raise RuntimeError("Manual IOM CSV must contain a unique 'id' column.")
                df = df_local

        if df.empty:
            logger.warning("No IOM data extracted from API or local file. Nothing to upload.")
            return

        # Integer coercion and invariant enforcement before upload
        df = _coerce_integer_like_columns(df)
        _assert_no_float_strings(df)

        # Save CSV snapshot
        out_path = PATHS["data_proc"] / OUTPUT_FILENAME
        df.to_csv(out_path, index=False)
        logger.info(f"Saved: {out_path} ({len(df):,} rows)")

        # Ensure table then upload
        _ensure_raw_table(engine)
        logger.info(f"Uploading {len(df):,} rows to DB {SCHEMA}.{TABLE_NAME}...")
        upload_to_postgis(engine, df, TABLE_NAME, SCHEMA, primary_keys=["id"])
        logger.info("IOM ingestion complete.")

    finally:
        client.close()


if __name__ == "__main__":
    main()
