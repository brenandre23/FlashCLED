"""
process_acled_hybrid.py (v3.0 - Dynamic Actor Scoring)
======================================================
Methodology: Semi-Supervised Semantic Projection with Dynamic Lethality-Based Actor Weighting

Pipeline:
1. ACTOR CALIBRATION: Query fatalities per actor over 1-year lookback, log1p transform, normalize
2. CONTRASTIVE MECHANISM DETECTION: Discriminative scoring using positive/negative anchors
3. UNCERTAINTY QUANTIFICATION: Confidence margins for ambiguous matches
4. ACTOR RISK WEIGHTING: Dynamic weights based on recent fatality data (no hardcoded weights)
5. TEXT QUALITY FILTERING: Downweight short/boilerplate notes

Output Features:
- mech_gold_pivot, mech_predatory_tax, mech_factional_infighting, mech_collective_punishment
- mech_*_uncertainty (confidence signals)
- acled_actor_risk_score (aggregated actor lethality from dynamic calibration)
- acled_combined_risk_score (mechanism × actor interaction)

PUBLICATION LAG: Config-driven via data.yaml acled_hybrid.publication_lag_steps.

Changes in v3.0:
- Removed hardcoded ACTOR_RISK_WEIGHTS (was based on stale IRR analysis)
- Added calibrate_actor_risk_model() for dynamic weight calculation
- Actor weights now derived purely from fatality data (log1p normalized)
- Actors with 0 fatalities in lookback window get weight 0.0 (fail-safe)
"""

import sys
import re
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text

# --- Setup ---
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
from utils import logger, get_db_engine, upload_to_postgis, SCHEMA, load_configs

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

# -----------------------------------------------------------------------------
# DYNAMIC ACTOR RISK CALIBRATION
# -----------------------------------------------------------------------------
# Actor risk weights are now calibrated dynamically from recent fatality data.
# This fallback is ONLY used if the database query returns empty (failsafe).

CALIBRATION_LOOKBACK_YEARS = 1  # Lookback window for fatality-based calibration

FALLBACK_RISK_WEIGHTS = {
    # Emergency fallback if DB query fails - based on historical CAR conflict actors
    "wagner group": 1.0,
    "upc: union for peace in the central african republic": 0.85,
    "anti-balaka": 0.80,
    "fprc: popular front for the renaissance of central africa": 0.75,
    "seleka militia": 0.70,
}

# -----------------------------------------------------------------------------
# CONTRASTIVE ANCHOR CONCEPTS (Mechanism Detector)
# -----------------------------------------------------------------------------
# Each concept defined by positive (target behavior) and negative (confounding behavior) anchors

ANCHOR_CONCEPTS = {
    "mech_gold_pivot": {
        "positive": [
            "artisanal gold mining site",
            "gold smuggling route",
            "gold ingot seizure",
            "dredging site attack",
            "gold convoy ambush",
            "control of gold mine",
            "illegal gold extraction",
            "gold miners killed"
        ],
        "negative": [
            "kimberley process certification",
            "legal mineral export",
            "industrial mining permit",
            "government mining revenue"
        ]
    },
    "mech_predatory_tax": {
        "positive": [
            "illegal roadblock",
            "extortion of traders",
            "protection racket",
            "levying taxes on goods",
            "checkpoint fee collection",
            "forced payment at barrier",
            "taxing market vendors",
            "toll collection by armed group",
            "convoy escort"
        ],
        "negative": [
            "customs checkpoint",
            "formal toll booth",
            "border control post",
            "police inspection",
            "veterinary control",
            "official tax collection"
        ]
    },
    "mech_factional_infighting": {
        "positive": [
            "clash between factions",
            "leadership dispute within group",
            "splinter group attack",
            "rival militia confrontation",
            "FPRC versus UPC",
            "internal rebel fighting",
            "defection and attack",
            "intra-group violence"
        ],
        "negative": [
            "joint military operation",
            "peace agreement signing",
            "coalition formation",
            "unified rebel command",
            "merger of armed groups",
            "coordinated attack on government"
        ]
    },
    "mech_collective_punishment": {
        "positive": [
            "burning of village",
            "reprisal against civilians",
            "punitive expedition",
            "accused of supporting rebels",
            "scorched earth tactics",
            "massacre of villagers",
            "collective retaliation",
            "cattle raiding reprisal",
            "targeted killing of fulani",
            "destruction of homes"
        ],
        "negative": [
            "clash with armed group",
            "attack on military base",
            "ambush of armed convoy",
            "battle for town control",
            "exchange of fire between combatants"
        ]
    }
}

# -----------------------------------------------------------------------------
# TEXT QUALITY PATTERNS
# -----------------------------------------------------------------------------
BOILERPLATE_PATTERNS = [
    r"no additional information",
    r"as reported",
    r"according to sources",
    r"details are unclear",
    r"unconfirmed reports",
    r"no further details"
]
BOILERPLATE_REGEX = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_device():
    """Select best available compute device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_publication_lag_days(data_cfg, features_cfg):
    """
    Calculate publication lag in days from config.
    
    Reads:
      - data.yaml:     acled_hybrid.publication_lag_steps (default: 1)
      - features.yaml: temporal.step_days (default: 14)
    """
    step_days = features_cfg.get('temporal', {}).get('step_days', 14)
    lag_steps = data_cfg.get('acled_hybrid', {}).get('publication_lag_steps', 1)
    lag_days = lag_steps * step_days
    
    logger.info(f"ACLED Hybrid publication lag: {lag_steps} steps × {step_days} days = {lag_days} days")
    return lag_days


def compute_text_quality_weight(text: str) -> float:
    """
    Compute quality weight for text based on length and boilerplate detection.
    
    Returns:
        float: Weight in [0.1, 1.0] where higher = better quality text
    """
    if not text or not isinstance(text, str):
        return 0.1
    
    text = text.strip()
    word_count = len(text.split())
    
    # Very short notes are unreliable
    if word_count < 5:
        return 0.1
    
    # Boilerplate detection
    if BOILERPLATE_REGEX.search(text):
        return 0.3
    
    # Scale by length (diminishing returns after 30 words)
    length_weight = min(1.0, word_count / 30)
    
    return max(0.1, length_weight)


def calibrate_actor_risk_model(engine) -> dict:
    """
    Dynamically calibrate actor risk weights from recent fatality data.
    
    Methodology:
    1. Query total fatalities per actor over the lookback window
    2. Filter to actors with >0 fatalities (lethal actors only)
    3. Apply log1p transform to compress scale
    4. Normalize so top killer = 1.0
    
    Returns:
        dict: Mapping of actor name (lowercase) to normalized risk weight [0, 1]
    """
    logger.info(f"Calibrating actor risk model (lookback={CALIBRATION_LOOKBACK_YEARS} year(s))...")
    
    query = f"""
        SELECT 
            LOWER(TRIM(actor1)) AS actor,
            SUM(fatalities) AS total_fatalities
        FROM {SCHEMA}.acled_events
        WHERE event_date >= CURRENT_DATE - INTERVAL '{CALIBRATION_LOOKBACK_YEARS} year'
          AND actor1 IS NOT NULL
          AND actor1 != ''
        GROUP BY LOWER(TRIM(actor1))
        HAVING SUM(fatalities) > 0
        ORDER BY total_fatalities DESC
    """
    
    try:
        df = pd.read_sql(query, engine)
        
        if df.empty:
            logger.warning("No lethal actors found in calibration window. Using fallback weights.")
            return FALLBACK_RISK_WEIGHTS.copy()
        
        # Log1p transform to compress scale (prevents extreme outliers from dominating)
        df['weight'] = np.log1p(df['total_fatalities'])
        
        # Normalize to [0, 1] where top killer = 1.0
        max_weight = df['weight'].max()
        if max_weight > 0:
            df['weight_normalized'] = df['weight'] / max_weight
        else:
            df['weight_normalized'] = 0.0
        
        # Build dictionary
        risk_weights = dict(zip(df['actor'], df['weight_normalized']))
        
        # Log top actors for diagnostics
        top_5 = df.nlargest(5, 'total_fatalities')[['actor', 'total_fatalities', 'weight_normalized']]
        logger.info(f"Calibrated {len(risk_weights)} lethal actors. Top 5:")
        for _, row in top_5.iterrows():
            logger.info(f"  {row['actor'][:50]}: {row['total_fatalities']} fatalities -> weight={row['weight_normalized']:.3f}")
        
        return risk_weights
        
    except Exception as e:
        logger.error(f"Actor calibration query failed: {e}. Using fallback weights.")
        return FALLBACK_RISK_WEIGHTS.copy()


def compute_actor_risk_score(actor1: str, actor2: str, risk_weights: dict) -> float:
    """
    Compute combined actor risk score from dynamically-calibrated weights.
    
    Args:
        actor1: Primary actor name (may be None)
        actor2: Secondary actor name (may be None)
        risk_weights: Dictionary mapping actor names (lowercase) to normalized weights [0, 1]
    
    Returns:
        float: Normalized risk score [0, 1] based on max actor risk.
               Returns 0.0 for actors with no fatalities in the calibration window.
    """
    scores = []
    
    for actor in [actor1, actor2]:
        if pd.isna(actor) or not actor:
            continue
        actor_lower = str(actor).lower().strip()
        
        # Exact match first
        if actor_lower in risk_weights:
            scores.append(risk_weights[actor_lower])
            continue
        
        # Partial match (e.g., "seleka" in "seleka militia faction")
        for known_actor, weight in risk_weights.items():
            if known_actor in actor_lower or actor_lower in known_actor:
                scores.append(weight)
                break
    
    if not scores:
        return 0.0  # Unknown actors or actors with 0 fatalities in lookback window
    
    # Return max risk (most dangerous actor present)
    return max(scores)


def compute_mechanism_scores_with_uncertainty(
    note_embeddings: np.ndarray,
    pos_embeddings: np.ndarray,
    neg_embeddings: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute discriminative mechanism scores with uncertainty quantification.
    
    Formula: Score = Clamp(Sim(Doc, Pos) - 0.5 * Sim(Doc, Neg), 0, 1)
    Uncertainty = 1 - (max_sim - second_max_sim) [margin-based]
    
    Args:
        note_embeddings: (N, D) array of document embeddings
        pos_embeddings: (P, D) array of positive anchor embeddings
        neg_embeddings: (Q, D) array of negative anchor embeddings
    
    Returns:
        scores: (N,) array of mechanism scores [0, 1]
        uncertainty: (N,) array of uncertainty values [0, 1]
    """
    # Compute similarities
    sim_pos = cosine_similarity(note_embeddings, pos_embeddings)  # (N, P)
    sim_neg = cosine_similarity(note_embeddings, neg_embeddings)  # (N, Q)
    
    # Discriminative score
    max_pos = np.max(sim_pos, axis=1)
    max_neg = np.max(sim_neg, axis=1)
    scores = np.clip(max_pos - (0.5 * max_neg), 0.0, 1.0)
    
    # Uncertainty: margin between best and second-best positive anchor
    if sim_pos.shape[1] > 1:
        sorted_pos = np.sort(sim_pos, axis=1)[:, ::-1]
        margin = sorted_pos[:, 0] - sorted_pos[:, 1]
    else:
        margin = np.ones(len(scores))
    
    uncertainty = np.clip(1.0 - margin, 0.0, 1.0)
    
    return scores, uncertainty


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_data(df: pd.DataFrame, risk_weights: dict) -> pd.DataFrame:
    """
    Process ACLED events through the full NLP pipeline.
    
    Args:
        df: DataFrame with ACLED events (must have notes, actor1, actor2 columns)
        risk_weights: Dynamically calibrated actor risk weights from calibrate_actor_risk_model()
    
    Steps:
    1. Clean text
    2. Compute text quality weights
    3. Compute actor risk scores (using dynamic weights)
    4. Compute mechanism scores with uncertainty
    5. Combine into final weighted scores
    """
    if df.empty:
        return df
    
    logger.info(f"Processing {len(df)} ACLED events...")
    
    # -------------------------------------------------------------------------
    # Step 1: Text Cleaning
    # -------------------------------------------------------------------------
    df['notes_clean'] = df['notes'].astype(str).fillna("").str.lower().str.strip()
    
    # -------------------------------------------------------------------------
    # Step 2: Text Quality Weights
    # -------------------------------------------------------------------------
    logger.info("Computing text quality weights...")
    df['text_quality_weight'] = df['notes_clean'].apply(compute_text_quality_weight)
    
    quality_stats = df['text_quality_weight'].describe()
    logger.info(f"  Quality weights: mean={quality_stats['mean']:.3f}, "
                f"min={quality_stats['min']:.3f}, max={quality_stats['max']:.3f}")
    
    # -------------------------------------------------------------------------
    # Step 3: Actor Risk Scores (using dynamic calibration)
    # -------------------------------------------------------------------------
    logger.info("Computing actor risk scores...")
    df['acled_actor_risk_score'] = df.apply(
        lambda x: compute_actor_risk_score(x.get('actor1'), x.get('actor2'), risk_weights), 
        axis=1
    )
    
    actor_coverage = (df['acled_actor_risk_score'] > 0).mean()
    logger.info(f"  Actor coverage: {actor_coverage:.1%} of events matched lethal actors")
    
    # -------------------------------------------------------------------------
    # Step 4: Mechanism Scores with Uncertainty
    # -------------------------------------------------------------------------
    device = get_device()
    logger.info(f"Loading transformer model ({MODEL_NAME}) on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    # Embed all notes
    logger.info("Embedding event notes...")
    note_embeddings = model.encode(
        df['notes_clean'].tolist(),
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    
    # Process each mechanism concept
    logger.info("Computing mechanism scores with uncertainty...")
    for concept, anchors in tqdm(ANCHOR_CONCEPTS.items(), desc="Mechanisms"):
        # Embed anchors
        pos_embeddings = model.encode(anchors["positive"], convert_to_numpy=True)
        neg_embeddings = model.encode(anchors["negative"], convert_to_numpy=True)
        
        # Compute scores and uncertainty
        scores, uncertainty = compute_mechanism_scores_with_uncertainty(
            note_embeddings, pos_embeddings, neg_embeddings
        )
        
        # Apply text quality weighting to scores
        weighted_scores = scores * df['text_quality_weight'].values
        
        # Store results
        df[concept] = weighted_scores
        df[f"{concept}_uncertainty"] = uncertainty
        
        # Log distribution
        nonzero_pct = (weighted_scores > 0.1).mean()
        logger.info(f"  {concept}: {nonzero_pct:.1%} events with score > 0.1, "
                    f"mean uncertainty: {uncertainty.mean():.3f}")
    
    # -------------------------------------------------------------------------
    # Step 5: Combined Risk Score (Mechanism × Actor Interaction)
    # -------------------------------------------------------------------------
    logger.info("Computing combined risk scores...")
    
    # Sum of all mechanism scores (represents "how predatory" the event is)
    mechanism_cols = list(ANCHOR_CONCEPTS.keys())
    df['acled_mechanism_intensity'] = df[mechanism_cols].sum(axis=1)
    
    # Combined score: mechanism intensity weighted by actor lethality
    # If actor_risk is 0 (unknown actor), fall back to mechanism score alone
    df['acled_combined_risk_score'] = np.where(
        df['acled_actor_risk_score'] > 0,
        df['acled_mechanism_intensity'] * (1 + df['acled_actor_risk_score']),
        df['acled_mechanism_intensity']
    )
    
    return df


def aggregate_and_upload(df: pd.DataFrame, engine, lag_days: int, configs: dict):
    """
    Aggregate scores by date/location and upload to database.
    INCLUDES FIX: Snaps dates to temporal spine defined in features.yaml.
    """
    # 1. Get Grid Settings from Configs
    step_days = configs['features']['temporal']['step_days']  # e.g., 14
    anchor_date_str = configs['data']['global_date_window']['start_date']  # e.g., "2000-01-01"
    anchor_date = pd.Timestamp(anchor_date_str)

    logger.info(f"Snapping events to {step_days}-day grid starting {anchor_date_str}...")

    # 2. Define Columns
    mechanism_cols = list(ANCHOR_CONCEPTS.keys())
    uncertainty_cols = [f"{c}_uncertainty" for c in mechanism_cols]
    derived_cols = ['acled_actor_risk_score', 'acled_mechanism_intensity', 'acled_combined_risk_score']

    feature_cols = mechanism_cols + uncertainty_cols + derived_cols

    # Ensure columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    # 3. SNAP-TO-GRID LOGIC
    # A. Apply Publication Lag first (Shift time forward)
    df['future_date'] = pd.to_datetime(df['event_date']) + pd.Timedelta(days=lag_days)

    # B. Calculate "Steps Since Anchor"
    days_diff = (df['future_date'] - anchor_date).dt.days
    step_indices = (days_diff // step_days).astype(int)

    # C. Project back to valid spine dates (start of each step window)
    df['snapped_date'] = anchor_date + pd.to_timedelta(step_indices * step_days, unit='D')

    logger.info(f"Aggregating {len(feature_cols)} features by snapped date/location...")

    # Aggregation rules
    agg_rules = {}
    for col in mechanism_cols:
        agg_rules[col] = 'sum'  # Accumulate within the 14-day window
    for col in uncertainty_cols:
        agg_rules[col] = 'mean'
    for col in derived_cols:
        agg_rules[col] = 'max' if 'risk' in col else 'sum'

    # Group by snapped date/location
    df_agg = df.groupby(['snapped_date', 'h3_index']).agg(agg_rules).reset_index()
    df_agg.rename(columns={'snapped_date': 'event_date'}, inplace=True)

    # Clean column names
    df_agg.columns = [c.replace(" ", "_") for c in df_agg.columns]
    
    # Create/recreate table
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.features_acled_hybrid;"))
        
        # Build column definitions
        col_defs = ['event_date DATE', 'h3_index BIGINT']
        for c in df_agg.columns:
            if c not in ['event_date', 'h3_index']:
                col_defs.append(f'"{c}" DOUBLE PRECISION')
        
        create_sql = f"""
            CREATE TABLE {SCHEMA}.features_acled_hybrid (
                {', '.join(col_defs)},
                PRIMARY KEY (event_date, h3_index)
            );
        """
        conn.execute(text(create_sql))
        conn.execute(text(
            f"CREATE INDEX IF NOT EXISTS idx_acled_hybrid_h3 ON {SCHEMA}.features_acled_hybrid (h3_index);"
        ))
        conn.execute(text(
            f"CREATE INDEX IF NOT EXISTS idx_acled_hybrid_date ON {SCHEMA}.features_acled_hybrid (event_date);"
        ))
    
    logger.info(f"Uploading {len(df_agg)} aggregated rows to database...")
    upload_to_postgis(
        engine, df_agg, "features_acled_hybrid", SCHEMA,
        primary_keys=['event_date', 'h3_index']
    )
    
    # Log summary statistics
    logger.info("Feature summary statistics:")
    for col in mechanism_cols[:4]:  # Just show first 4
        stats = df_agg[col].describe()
        logger.info(f"  {col}: mean={stats['mean']:.4f}, max={stats['max']:.4f}")


def validate_output(df: pd.DataFrame) -> bool:
    """
    Validate processed data before upload.
    """
    logger.info("Running validation checks...")
    
    required_cols = set(ANCHOR_CONCEPTS.keys())
    missing = required_cols - set(df.columns)
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return False
    
    # Check value ranges
    for col in ANCHOR_CONCEPTS.keys():
        if not df[col].between(-0.001, 1.001).all():
            logger.warning(f"Values out of range for {col}: "
                          f"min={df[col].min():.4f}, max={df[col].max():.4f}")
    
    # Check for all-zero columns
    for col in ANCHOR_CONCEPTS.keys():
        if df[col].sum() == 0:
            logger.warning(f"Column {col} is all zeros - check anchor definitions")
    
    logger.info("Validation passed")
    return True


# =============================================================================
# ENTRY POINT
# =============================================================================

def run(configs=None):
    """
    Main entry point for ACLED Hybrid NLP processing.
    """
    engine = get_db_engine()
    
    # Load configs
    if configs is None:
        cfgs = load_configs()
        if isinstance(cfgs, tuple):
            configs = {"data": cfgs[0], "features": cfgs[1], "models": cfgs[2]}
        else:
            configs = cfgs
    
    data_cfg = configs.get("data", {})
    features_cfg = configs.get("features", {})
    
    # Get publication lag
    lag_days = get_publication_lag_days(data_cfg, features_cfg)
    
    # Calibrate actor risk model BEFORE fetching event data
    risk_weights = calibrate_actor_risk_model(engine)
    
    # Fetch data with actor columns
    # Filter to precision levels 1-2 for reliable H3 spatial assignment
    query = """
        SELECT
            event_date,
            h3_index,
            notes,
            actor1,
            actor2
        FROM car_cewp.acled_events
        WHERE notes IS NOT NULL
          AND geo_precision IN (1, 2)
          AND time_precision IN (1, 2)
    """
    
    try:
        logger.info("Fetching ACLED events from database...")
        df = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(df)} ACLED events with notes")
        
        if df.empty:
            logger.warning("No ACLED events found with notes")
            return
        
        # Process (with dynamic risk weights)
        df_processed = process_data(df, risk_weights)
        
        # Validate
        if not validate_output(df_processed):
            raise ValueError("Validation failed")
        
        # Upload
        aggregate_and_upload(df_processed, engine, lag_days, configs)
        
        logger.info("ACLED Hybrid processing complete")
        
    except Exception as e:
        logger.error(f"ACLED Hybrid processing failed: {e}")
        raise


if __name__ == "__main__":
    run()