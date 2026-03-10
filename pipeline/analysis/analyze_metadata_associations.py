"""
analyze_metadata_associations.py
================================
Purpose:
  Correlate 'Latent Topics' (CrisisWatch & ACLED NLP) with Model Predictions.
  
  Answers:
  1. "Which narratives (e.g., 'Peace Treaty', 'Rebel Advance') drive the highest fatality forecasts?"
  2. "Are we over-predicting violence in areas with benign topics?"
"""

import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sqlalchemy import text

# --- Setup ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, load_configs

OUTPUT_DIR = ROOT_DIR / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_predictions(engine):
    """Fetches the latest predictions (14d horizon)."""
    logger.info("Fetching predictions from car_cewp.predictions (horizon='14d', learner='xgboost')...")
    try:
        query = text("""
            SELECT 
                date,
                h3_index,
                conflict_prob,
                predicted_fatalities,
                risk_score,
                horizon,
                learner
            FROM car_cewp.predictions
            WHERE horizon = '14d'
              AND learner = 'xgboost'
              AND date >= NOW() - INTERVAL '3 months'
        """)
        return pd.read_sql(query, engine)
    except Exception as e:
        logger.warning(f"Could not load predictions: {e}")
        return pd.DataFrame()


def load_crisiswatch_topics(engine):
    """Fetches CrisisWatch semantic topics and maps IDs to names."""
    # Mapping based on process_crisiswatch.py CONCEPTS
    cw_id_to_name = {
        10: "Pillar 10: Parallel Governance",
        11: "Pillar 11: Transnational Predation",
        12: "Pillar 12: Guerrilla Fragmentation",
        13: "Pillar 13: Ethno-Pastoral Rupture",
        99: "Narrative Velocity (National)"
    }
    try:
        query = """
            SELECT date, h3_index, cw_topic_id, spatial_confidence_norm as topic_intensity
            FROM car_cewp.features_crisiswatch
        """
        df = pd.read_sql(query, engine)
        if not df.empty:
            df['cw_topic_name'] = df['cw_topic_id'].map(cw_id_to_name)
        return df
    except Exception as e:
        logger.warning(f"Could not load CrisisWatch topics: {e}")
        return pd.DataFrame()


def load_acled_topics(engine):
    """Fetches ACLED micro-dynamics from the hybrid NLP table."""
    try:
        # The features_acled_hybrid table is wide, with columns for each mechanism
        query = """
            SELECT event_date as date, h3_index, 
                   mech_gold_pivot, mech_predatory_tax, 
                   mech_factional_infighting, mech_collective_punishment
            FROM car_cewp.features_acled_hybrid
        """
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return df
            
        # Melt the wide table into long format for analyze_impact
        mechs = ["mech_gold_pivot", "mech_predatory_tax", "mech_factional_infighting", "mech_collective_punishment"]
        df_melted = df.melt(
            id_vars=['date', 'h3_index'], 
            value_vars=mechs,
            var_name='acled_topic_name', 
            value_name='topic_intensity'
        )
        
        # Filter to meaningful signals
        df_melted = df_melted[df_melted['topic_intensity'] > 0.1]
        
        return df_melted
    except Exception as e:
        logger.warning(f"Could not load ACLED topics: {e}")
        return pd.DataFrame()


def analyze_impact(preds, features, feature_col, feature_type):
    """Calculates average predicted fatalities per topic, weighted by intensity."""
    logger.info(f"Analyzing impact of {feature_type}...")
    
    # Ensure date/h3 match types
    if preds.empty or features.empty:
        return
        
    preds['date'] = pd.to_datetime(preds['date'])
    features['date'] = pd.to_datetime(features['date'])
    
    merged = pd.merge(preds, features, on=['date', 'h3_index'], how='inner')
    
    if merged.empty:
        logger.warning(f"No overlap found between predictions and {feature_type}.")
        return

    # Weight predicted fatalities by topic intensity
    merged['weighted_fatalities'] = merged['predicted_fatalities'] * merged['topic_intensity']
    
    impact = merged.groupby(feature_col).agg({
        'predicted_fatalities': 'mean',
        'topic_intensity': 'mean',
        'h3_index': 'count'
    }).reset_index()
    
    impact.columns = [feature_col, 'mean_predicted_fatalities', 'mean_intensity', 'occurrence_count']
    impact = impact.sort_values(by='mean_predicted_fatalities', ascending=False)
    impact = impact[impact['occurrence_count'] > 5] # Lowered threshold
    
    if impact.empty:
        logger.warning(f"No {feature_type} groups exceed the frequency threshold.")
        return

    print(f"\n--- {feature_type} Impact on Predictions ---")
    print(impact.head(10))
    
    plt.figure(figsize=(12, 7))
    sns.barplot(data=impact.head(10), x='mean_predicted_fatalities', y=feature_col, palette='magma')
    plt.title(f"Average Predicted Fatalities by {feature_type}")
    plt.xlabel("Avg. Predicted Fatalities (14d Forecast)")
    plt.tight_layout()
    
    out_file = OUTPUT_DIR / f"impact_{feature_type.lower().replace(' ', '_')}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    logger.info(f"Saved plot to {out_file}")


def main():
    engine = get_db_engine()
    
    df_pred = load_predictions(engine)
    if df_pred.empty:
        logger.warning("Prediction table is empty; skipping metadata association analysis.")
        return

    df_cw = load_crisiswatch_topics(engine)
    if not df_cw.empty:
        analyze_impact(df_pred, df_cw, 'cw_topic_name', "CrisisWatch Narrative")
    else:
        logger.warning("CrisisWatch topics table empty/missing.")

    df_acled = load_acled_topics(engine)
    if not df_acled.empty:
        analyze_impact(df_pred, df_acled, 'acled_topic_name', "ACLED Micro-Tactic")
    else:
        logger.warning("ACLED NLP topics table empty/missing.")


if __name__ == "__main__":
    main()
