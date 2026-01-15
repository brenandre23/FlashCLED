"""
fetch_gdelt_themes.py
=====================
Fetch national-level GDELT theme counts for the configured top-300 list.
"""
import sys
import importlib
import pandas as pd
from google.cloud import bigquery
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
from utils import logger, get_db_engine, upload_to_postgis, SCHEMA, load_configs  # noqa: E402

load_dotenv(ROOT_DIR / ".env")

# Load Target Themes
THEMES_FILE = ROOT_DIR / "data" / "processed" / "gdelt_top_300_themes.csv"

def _config_dates():
    try:
        cfg = load_configs()
        data_cfg = cfg[0] if isinstance(cfg, tuple) else getattr(cfg, "data", {})
        gw = (data_cfg or {}).get("global_date_window", {})
        return gw.get("start_date"), gw.get("end_date")
    except Exception as e:
        logger.warning(f"Could not read date window from data.yaml: {e}")
        return None, None


def _load_target_themes():
    return pd.read_csv(THEMES_FILE)["Theme"].tolist()


def run(configs=None, engine=None):
    cfg_start, cfg_end = _config_dates()
    # Self-initialize theme list if missing
    if not THEMES_FILE.exists():
        logger.warning("Config missing, running discovery...")
        discover = importlib.import_module("pipeline.ingestion.discover_gdelt_themes")
        discover.main()

    target_themes = _load_target_themes()
    if not target_themes:
        logger.error("No target themes found in gdelt_top_300_themes.csv")
        return

    query = """
    SELECT
      CAST(DATE(_PARTITIONDATE) AS STRING) as date,
      theme,
      COUNT(*) as count
    FROM `gdelt-bq.gdeltv2.gkg_partitioned`,
    UNNEST(SPLIT(V2Themes, ';')) as theme
    WHERE _PARTITIONDATE >= @start_date
      AND DATE(_PARTITIONDATE) <= @end_date
      AND (
            REGEXP_CONTAINS(V2Locations, r"#CT#")
         OR REGEXP_CONTAINS(V2Locations, r"#CF#")
         OR REGEXP_CONTAINS(V2Locations, r"#CAF#")
      )
      AND theme IN UNNEST(@themes)
    GROUP BY 1, 2
    """

    logger.info("dYs? Starting GDELT Themes Ingestion...")
    client = bigquery.Client()

    # 1. Fetch
    logger.info("   Querying BigQuery (National Context)...")
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("start_date", "DATE", cfg_start or "2015-02-18"),
            bigquery.ScalarQueryParameter("end_date", "DATE", cfg_end or pd.Timestamp.now().date()),
            bigquery.ArrayQueryParameter("themes", "STRING", target_themes),
        ]
    )
    df = client.query(query, job_config=job_config).to_dataframe()

    # Fallback: if filtered list returns empty, pull top 300 themes for CAR directly
    if df.empty:
        logger.warning("No rows returned for target themes; computing top themes directly for CAR.")
        fallback_query = """
        WITH filtered AS (
          SELECT
            DATE(_PARTITIONDATE) as date,
            theme
          FROM `gdelt-bq.gdeltv2.gkg_partitioned`,
          UNNEST(SPLIT(V2Themes, ';')) as theme
          WHERE _PARTITIONDATE >= @start_date
            AND DATE(_PARTITIONDATE) <= @end_date
            AND (
                  REGEXP_CONTAINS(V2Locations, r"#CT#")
               OR REGEXP_CONTAINS(V2Locations, r"#CF#")
               OR REGEXP_CONTAINS(V2Locations, r"#CAF#")
            )
        ),
        top_themes AS (
          SELECT theme
          FROM filtered
          GROUP BY theme
          ORDER BY COUNT(*) DESC
          LIMIT 300
        )
        SELECT CAST(date AS STRING) as date, f.theme, COUNT(*) as count
        FROM filtered f
        JOIN top_themes t ON f.theme = t.theme
        GROUP BY 1, 2
        """
        df = client.query(
            fallback_query,
            job_config=bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("start_date", "DATE", cfg_start or "2015-02-18"),
                    bigquery.ScalarQueryParameter("end_date", "DATE", cfg_end or pd.Timestamp.now().date()),
                ]
            ),
        ).to_dataframe()

    # 2. Pivot (Long -> Wide)
    logger.info("   Pivoting to Wide Format...")
    df_wide = df.pivot_table(index="date", columns="theme", values="count", fill_value=0)
    df_wide.columns = [f"theme_{c.lower().replace('.', '')}_count" for c in df_wide.columns]
    df_wide.reset_index(inplace=True)
    df_wide["date"] = pd.to_datetime(df_wide["date"])

    # 3. Upload
    engine = engine or get_db_engine()
    upload_to_postgis(engine, df_wide, "features_national_daily", SCHEMA, primary_keys=["date"])
    logger.info(f"ƒo. Context Loaded: {len(df_wide)} days.")


if __name__ == "__main__":
    run()
