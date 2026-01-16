"""
process_acled_hybrid.py (PhD Edition)
=====================================
Methodology: Semi-Supervised Semantic Projection
1. ENSEMBLE SCORING: Detects conflict drivers using a weighted average of:
   - Explicit Regex (Precision)
   - Latent Vector Similarity (Recall via Anchor Concepts)
2. RESIDUAL CLUSTERING: Clusters the 'Context' of events, automatically 
   labeling topics based on TF-IDF centroids.

UPDATES:
- PUBLICATION LAG: Config-driven via data.yaml acled_hybrid.publication_lag_steps.
  ACLED has ~6 day release cycle (1 step × 14 days = 14 days).

Output: Continuous risk scores (0.0-1.0) rather than binary flags.
"""

import sys
import re
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text

# --- Setup ---
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
from utils import logger, get_db_engine, upload_to_postgis, SCHEMA, load_configs

# --- CONFIG ---
MODEL_NAME = "all-MiniLM-L6-v2"  # Fast, efficient sentence transformer
N_CLUSTERS = 8 
BATCH_SIZE = 64

# --- DRIVER DEFINITIONS (Hybrid) ---
# Each driver has Regex (Hard match) and Anchor Sentences (Soft match)
DRIVERS = {
    "driver_resource_cattle": {
        "regex": r"\b(cattle|cow|herder|grazing|pasture|transhumance|livestock|breeder|fulani|anti-balaka)\b",
        "anchors": [
            "Violence involving cattle herders and farmers over grazing land.",
            "Armed pastoralists attacked the village stealing livestock.",
            "Transhumance migration corridors blocked by armed groups."
        ]
    },
    "driver_resource_mining": {
        "regex": r"\b(gold|diamond|mine|mining|pit|artisan|quarry|mineral|bria|bambari)\b",
        "anchors": [
            "Armed groups fighting for control of gold and diamond mines.",
            "Illicit mining activities and taxation by rebels.",
            "Artisanal miners attacked at the quarry site."
        ]
    },
    "driver_econ_taxation": {
        "regex": r"\b(tax|checkpoint|roadblock|extort|loot|pillage|money|fee|barrier|trade)\b",
        "anchors": [
            "Rebels set up roadblocks to extort money from travelers.",
            "Illegal taxation and looting of commercial trucks.",
            "Pillaging of local markets and shops by armed men."
        ]
    },
    "driver_political_coup": {
        "regex": r"\b(coup|overthrow|arrest|detain|president|minister|election|vote|decree)\b",
        "anchors": [
            "Attempted coup d'etat against the government.",
            "Arrest of opposition leaders and political instability.",
            "Protests regarding the presidential election results."
        ]
    },
    "driver_civilian_abuse": {
        "regex": r"\b(abduct|kidnap|rape|sexual|torture|behead|execution|civilian)\b",
        "anchors": [
            "Targeted violence and human rights abuses against civilians.",
            "Kidnapping of villagers for ransom.",
            "Sexual violence and torture by armed combatants."
        ]
    }
}


def get_publication_lag_days(data_cfg, features_cfg):
    """
    Calculate publication lag in days from config.
    
    Reads:
      - data.yaml:     acled_hybrid.publication_lag_steps (default: 1)
      - features.yaml: temporal.step_days (default: 14)
    
    Returns:
      int: Number of days to shift stored dates forward
    """
    step_days = features_cfg.get('temporal', {}).get('step_days', 14)
    lag_steps = data_cfg.get('acled_hybrid', {}).get('publication_lag_steps', 1)
    lag_days = lag_steps * step_days
    
    logger.info(f"ACLED Hybrid publication lag: {lag_steps} steps × {step_days} days = {lag_days} days")
    return lag_days


# -----------------------------------------------------------------------------
# 1. CORE LOGIC
# -----------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available(): return "cuda"
    elif torch.backends.mps.is_available(): return "mps"
    return "cpu"


def compute_soft_scores(model, texts, driver_anchors):
    """
    Computes Semantic Similarity (0.0 - 1.0) between events and Driver Anchors.
    """
    # Embed Events
    event_embeddings = model.encode(texts, batch_size=BATCH_SIZE, convert_to_numpy=True, show_progress_bar=False)
    
    # Embed Anchors
    anchor_embeddings = model.encode(driver_anchors, convert_to_numpy=True)
    
    # Cosine Similarity (Events x Anchors)
    # We take the MAX similarity to any of the anchors for that driver
    sim_matrix = cosine_similarity(event_embeddings, anchor_embeddings)
    scores = np.max(sim_matrix, axis=1)  # Max across anchors
    
    return scores, event_embeddings


def get_cluster_labels(df, vectorizer, n_top_words=3):
    """
    Generates semantic names for clusters (e.g., 'theme_rebel_clash')
    using TF-IDF centroid analysis.
    """
    labels_map = {}
    unique_clusters = sorted(df['cluster_id'].unique())
    
    logger.info("🏷️  Auto-labeling latent clusters...")
    
    for label in unique_clusters:
        # Get all text in this cluster
        cluster_text = df[df['cluster_id'] == label]['notes_clean']
        if len(cluster_text) < 5: 
            labels_map[label] = f"theme_minor_{label}"
            continue
            
        # Fit TF-IDF on this cluster
        tfidf = vectorizer.fit_transform(cluster_text)
        feature_names = np.array(vectorizer.get_feature_names_out())
        
        # Average TF-IDF score per word
        avg_scores = np.mean(tfidf.toarray(), axis=0)
        top_indices = avg_scores.argsort()[-n_top_words:][::-1]
        top_words = feature_names[top_indices]
        
        # Create slug
        slug = "_".join(top_words)
        labels_map[label] = f"theme_ctx_{slug}"
        
    return labels_map


# -----------------------------------------------------------------------------
# 2. PIPELINE
# -----------------------------------------------------------------------------

def process_data(df):
    if df.empty: return df

    # A. Cleaning (Keep syntax for BERT, but lower case)
    df['notes_clean'] = df['notes'].astype(str).fillna("").str.lower().str.strip()
    
    # B. Init Model
    device = get_device()
    logger.info(f"🧠 Loading Transformer ({MODEL_NAME}) on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    # C. Calculate Driver Scores (The Hybrid Ensemble)
    logger.info("⚖️  Calculating Hybrid Driver Scores...")
    
    all_embeddings = None
    
    for driver, config in tqdm(DRIVERS.items(), desc="Drivers"):
        # 1. Regex Score (Binary 0 or 1)
        regex_score = df['notes_clean'].str.contains(config['regex'], regex=True).astype(float)
        
        # 2. Semantic Score (Continuous 0.0 to 1.0)
        semantic_score, embeddings = compute_soft_scores(model, df['notes_clean'].tolist(), config['anchors'])
        
        # Cache embeddings on first run to avoid re-computing
        if all_embeddings is None: all_embeddings = embeddings
            
        # 3. Ensemble (Soft Voting)
        final_score = np.maximum(regex_score, semantic_score)
        
        # 4. Filter Noise (Clip low semantic matches that aren't regex matches)
        final_score = np.where((regex_score == 0) & (final_score < 0.25), 0.0, final_score)
        
        df[driver] = final_score

    # D. Residual Clustering (The "Context")
    logger.info("🔍 Clustering Contextual Themes...")
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, batch_size=256)
    df['cluster_id'] = kmeans.fit_predict(all_embeddings)
    
    # E. Dynamic Labeling
    tfidf_vec = TfidfVectorizer(stop_words='english', max_features=1000)
    cluster_names = get_cluster_labels(df, tfidf_vec)
    
    # Map IDs to Names and One-Hot Encode
    df['cluster_name'] = df['cluster_id'].map(cluster_names)
    dummies = pd.get_dummies(df['cluster_name'])
    df = pd.concat([df, dummies], axis=1)

    return df


def aggregate_and_upload(df, engine, lag_days):
    """
    Aggregate driver scores and upload with publication lag applied.
    
    lag_days: Number of days to shift stored date forward (from config)
    """
    # Select columns
    driver_cols = list(DRIVERS.keys())
    theme_cols = [c for c in df.columns if c.startswith('theme_ctx_')]
    
    feature_cols = driver_cols + theme_cols
    
    logger.info(f"📉 Aggregating {len(feature_cols)} features (Summing Probabilities)...")
    
    # Group by Date/Location
    df_agg = df.groupby(['event_date', 'h3_index'])[feature_cols].sum().reset_index()
    
    # PUBLICATION LAG: Shift event_date forward by lag_days
    # Events occurring on 'event_date' become "available" at (event_date + lag_days)
    df_agg['event_date'] = pd.to_datetime(df_agg['event_date']) + pd.Timedelta(days=lag_days)
    
    logger.info(f"Applied {lag_days}-day publication lag to event dates.")
    
    # Rename for DB safety (truncate long generated names if needed)
    df_agg.columns = [c.replace(" ", "_") for c in df_agg.columns]

    # Recreate target table to align schema with current columns
    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(text(f"DROP TABLE IF EXISTS {SCHEMA}.features_acled_hybrid;"))
        col_defs = ['event_date DATE', 'h3_index BIGINT']
        col_defs += [f'"{c}" DOUBLE PRECISION' for c in df_agg.columns if c not in ['event_date', 'h3_index']]
        create_sql = f"""
            CREATE TABLE {SCHEMA}.features_acled_hybrid (
                {', '.join(col_defs)},
                PRIMARY KEY (event_date, h3_index)
            );
        """
        conn.execute(text(create_sql))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_features_acled_hybrid_h3 ON {SCHEMA}.features_acled_hybrid (h3_index);"))
    
    logger.info(f"💾 Uploading {len(df_agg)} rows to DB...")
    upload_to_postgis(engine, df_agg, "features_acled_hybrid", "car_cewp", 
                     primary_keys=['event_date', 'h3_index'])


def run(configs=None, engine=None):
    engine = engine or get_db_engine()
    
    # Load configs if not provided
    if configs is None:
        cfgs = load_configs()
        if isinstance(cfgs, tuple):
            configs = {"data": cfgs[0], "features": cfgs[1], "models": cfgs[2]}
        else:
            configs = cfgs
    
    data_cfg = configs.get("data", {})
    features_cfg = configs.get("features", {})
    
    # Get publication lag from config
    lag_days = get_publication_lag_days(data_cfg, features_cfg)
    
    # Fetch Data (Standard ACLED fetch)
    query = """
        SELECT event_date, h3_index, notes 
        FROM car_cewp.acled_events 
        WHERE notes IS NOT NULL
    """
    
    try:
        df = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(df)} ACLED notes.")
        
        if not df.empty:
            df_processed = process_data(df)
            aggregate_and_upload(df_processed, engine, lag_days)
            
    except Exception as e:
        logger.error(f"ACLED Hybrid process failed: {e}")
        raise e


if __name__ == "__main__":
    run()
