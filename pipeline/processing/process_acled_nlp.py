"""
process_acled_nlp.py
====================
Purpose: 
  1. Extract LATENT SIGNALS from ACLED 'notes' (Micro-dynamics).
  2. Use ConfliBERT + BERTopic to cluster events (e.g., "Roadblock Extortion", "Wagner Execution").
  3. SPATIAL BROADCAST: Apply H3 Ripple (k=1) to diffuse the tactical risk to neighbor cells.
  
Outcome: 
  Instead of just "Count of Battles", the model sees "Count of Wagner-specific Executions".
"""

import sys
import pandas as pd
import numpy as np
import h3.api.basic_int as h3
from pathlib import Path
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

# --- Setup ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, load_configs, upload_to_postgis

SCHEMA = "car_cewp"
INPUT_TABLE = "acled_events"
OUTPUT_TABLE = "features_nlp_acled"

# State-of-the-Art Conflict Embeddings
EMBEDDING_MODEL = "snowood1/ConfliBERT-scr-uncased"

def fetch_acled_notes(engine):
    """
    Fetch notes for the entire history to build a consistent Topic Model.
    """
    logger.info("Fetching ACLED notes from database...")
    query = f"""
        SELECT 
            event_id_cnty, 
            event_date, 
            h3_index, 
            notes 
        FROM {SCHEMA}.{INPUT_TABLE}
        WHERE notes IS NOT NULL AND TRIM(notes) != ''
    """
    return pd.read_sql(query, engine)

def apply_h3_ripple(df, resolution):
    """
    Spatial Smoothing:
    If a specific tactical event happens in Cell A, Cell B (neighbor) is also at risk.
    We broadcast the event with a decay factor.
    """
    logger.info("Applying Spatial Ripple (Risk Diffusion)...")
    
    expanded_rows = []
    
    # Pre-calculate ring weights
    # Ring 0 (Epicenter): 1.0
    # Ring 1 (Neighbor):  0.5
    decay_map = {0: 1.0, 1: 0.5}
    
    for row in df.itertuples():
        center_h3 = int(row.h3_index)
        
        # Get 1-ring neighbors (k=1)
        # grid_disk_dist returns (h3_index, distance) tuples
        try:
            rings = h3.grid_disk_dist(center_h3, 1)
        except:
            continue # Skip invalid H3
            
        for cell, dist in rings:
            if dist in decay_map:
                expanded_rows.append({
                    "date": row.event_date,
                    "h3_index": cell,
                    "acled_topic_id": row.topic_id,
                    "risk_contribution": decay_map[dist] # 1.0 or 0.5
                })
                
    return pd.DataFrame(expanded_rows)

def run(configs, engine):
    # 1. Load Data
    df = fetch_acled_notes(engine)
    if df.empty:
        logger.warning("No ACLED notes found. Skipping NLP.")
        return

    # 2. Embed & Cluster (The "Brain")
    logger.info(f"--- Training ConfliBERT Topic Model on {len(df)} events ---")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    
    # We use n_gram_range=(1, 2) to capture "Cattle Theft" or "Peace Deal"
    topic_model = BERTopic(
        embedding_model=embedding_model,
        verbose=True,
        n_gram_range=(1, 2), 
        min_topic_size=30, # High granularity: detect small, specific tactics
        calculate_probabilities=False 
    )
    
    # Fit the model
    notes_list = df['notes'].tolist()
    topics, _ = topic_model.fit_transform(notes_list)
    df['topic_id'] = topics

    # 3. Save Topic Definitions (Context for the Analyst)
    # This allows you to inspect: "Topic 4 = Wagner/Russian/Mercenary"
    topic_info = topic_model.get_topic_info()
    out_path = configs['paths']['data_proc'] / "acled_topic_definitions.csv"
    topic_info.to_csv(out_path, index=False)
    logger.info(f"Saved topic definitions to {out_path}")

    # 4. Filter Noise
    # Topic -1 is "Outlier/Noise" in BERTopic. We drop it to reduce database bloat.
    df_clean = df[df['topic_id'] != -1].copy()

    # 5. Spatial Broadcasting (The "Ripple")
    # This transforms 1 Point Event -> 7 Area Risk Signals
    resolution = configs['features']['spatial']['h3_resolution']
    df_rippled = apply_h3_ripple(df_clean, resolution)
    
    # 6. Aggregation
    # Multiple events might ripple into the same cell on the same day.
    # We sum the risk scores.
    df_final = df_rippled.groupby(['date', 'h3_index', 'acled_topic_id'])['risk_contribution'].sum().reset_index()
    df_final.rename(columns={'risk_contribution': 'topic_intensity'}, inplace=True)

    # 7. Upload
    logger.info(f"Uploading {len(df_final)} NLP features (Rippled) to DB...")
    upload_to_postgis(
        engine, 
        df_final, 
        OUTPUT_TABLE, 
        SCHEMA, 
        primary_keys=['h3_index', 'date', 'acled_topic_id']
    )
    logger.info("ACLED NLP Processing Complete.")

if __name__ == "__main__":
    cfg = load_configs()
    eng = get_db_engine()
    run(cfg if isinstance(cfg, dict) else cfg[0], eng)