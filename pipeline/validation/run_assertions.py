"""
Data contract assertions for CEWP pipeline.
Run this before building the feature matrix to catch schema/data issues early.
"""
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
from utils import logger, get_db_engine  # noqa: E402


ASSERTIONS = [
    {
        "name": "Assertion 1: H3 Types",
        "sql": """
            SELECT table_name FROM information_schema.columns
            WHERE table_schema = 'car_cewp' AND column_name = 'h3_index'
              AND data_type NOT IN ('bigint', 'int8')
        """,
        "fail_msg": "H3 columns detected as non-BIGINT (likely Text/Float). Fix schema."
    },
    {
        "name": "Assertion 3: Missing Static Features",
        "sql": """
            SELECT count(*) FROM car_cewp.features_static
            WHERE dist_to_road IS NULL OR dist_to_city IS NULL OR elevation_mean IS NULL
        """,
        "fail_msg": "Static features contain NULLs. Run calculate_static_distances.py again."
    },
    {
        "name": "Assertion 4: Distance values are valid",
        "sql": """
            SELECT MIN(dist_to_road) as min_road, MAX(dist_to_road) as max_road
            FROM car_cewp.features_static
        """,
        "fail_msg": "Distance values out of expected range (min < 0 or max > 500km)."
    },
    {
        "name": "Assertion 5: Temporal Gaps",
        "sql": """
            WITH gaps AS (
                SELECT date, LEAD(date) OVER (ORDER BY date) - date as diff
                FROM (SELECT DISTINCT date FROM car_cewp.temporal_features) d
            )
            SELECT count(*) FROM gaps WHERE EXTRACT(day FROM diff) > 15
        """,
        "fail_msg": "Found gaps in temporal spine larger than step size."
    },
    # Note: Assertion 8 handled via code below (prevalence)
]


def check_prevalence(conn):
    """Assertion 8: Target prevalence (allow very sparse grids)."""
    # Use fatalities_14d_sum as proxy for future targets
    prev_sql = """
        SELECT
          SUM(CASE WHEN fatalities_14d_sum > 0 THEN 1 ELSE 0 END)::float / COUNT(*) AS prevalence
        FROM car_cewp.temporal_features
    """
    prevalence = conn.execute(text(prev_sql)).scalar()
    if prevalence is None:
        raise RuntimeError("Could not compute prevalence from temporal_features.")
    if prevalence <= 0.0001:
        raise AssertionError(f"Assertion 8: prevalence {prevalence:.6%} is suspiciously low (<= 0.01%).")
    logger.info(f"✅ Assertion 8: prevalence = {prevalence:.4%}")


def run_checks():
    engine = get_db_engine()
    all_passed = True

    print("\n🔎 RUNNING DATA CONTRACT VALIDATION...")

    with engine.connect() as conn:
        for check in ASSERTIONS:
            try:
                result = conn.execute(text(check["sql"])).fetchone()
                failed = False

                if check["name"].startswith("Assertion 4"):
                    min_road, max_road = result
                    failed = (min_road is not None and min_road < 0) or (max_road is not None and max_road > 500000)
                    result_display = {"min_road": min_road, "max_road": max_road}
                else:
                    val = result[0] if result is not None else None
                    failed = bool(val)
                    result_display = val

                if failed:
                    logger.error(f"❌ FAILED: {check['name']}")
                    logger.error(f"   Reason: {check['fail_msg']}")
                    logger.error(f"   Result Value: {result_display}")
                    all_passed = False
                else:
                    print(f"   ✅ PASS: {check['name']}")
            except Exception as e:
                logger.error(f"   ⚠️ ERROR executing {check['name']}: {e}")
                all_passed = False
                try:
                    conn.rollback()
                except Exception:
                    pass

    # Assertion 8: Prevalence check (tolerant floor) in a fresh connection
    try:
        with engine.connect() as conn:
            check_prevalence(conn)
    except Exception as e:
        logger.error(f"❌ FAILED: Assertion 8 (Prevalence) -> {e}")
        all_passed = False

    if not all_passed:
        print("\n⛔ PIPELINE HALTED. Fix data contracts before training.\n")
        sys.exit(1)
    else:
        print("\n✨ ALL SYSTEMS GO.\n")


if __name__ == "__main__":
    run_checks()
