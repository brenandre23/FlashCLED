"""
interpret_crisiswatch_topics.py
===============================
Decodes the "Black Box" of your PCA features.
1. Loads the raw text and the embeddings.
2. Re-runs PCA (identical to the processing step).
3. Prints the top 5 representative sentences for each of the 16 topics.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

# --- Setup ---
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
from utils import logger

N_COMPONENTS = 16

def interpret_topics():
    # 1. Load Data
    text_path = ROOT_DIR / "pipeline" / "processing" / "crisiswatch_parsed_semantic.csv"
    emb_path = ROOT_DIR / "pipeline" / "processing" / "crisiswatch_embeddings.parquet"
    
    if not text_path.exists() or not emb_path.exists():
        logger.error("❌ Missing input files. Run the parsing and embedding steps first.")
        return

    logger.info("Loading Text and Embeddings...")
    df_text = pd.read_csv(text_path)
    df_emb = pd.read_parquet(emb_path)

    # Sanity Check: Ensure they align
    if len(df_text) != len(df_emb):
        logger.error(f"❌ Mismatch! Text has {len(df_text)} rows, Embeddings has {len(df_emb)}.")
        return

    # 2. Fit PCA
    logger.info(f"Fitting PCA (n={N_COMPONENTS})...")
    matrix = np.stack(df_emb['embedding'].values)
    
    pca = PCA(n_components=N_COMPONENTS)
    # Result is a matrix of shape (n_samples, 16)
    topics_matrix = pca.fit_transform(matrix)
    
    # 3. Interpret Each Topic
    logger.info("🔍 TOPIC INTERPRETATION REPORT")
    print("="*60)
    
    for topic_idx in range(N_COMPONENTS):
        # Get the scores for this specific topic across all sentences
        scores = topics_matrix[:, topic_idx]
        
        # Get indices of the top 5 sentences that "lit up" this neuron the most
        # argsort gives ascending, so we take the last 5 and reverse them
        top_indices = scores.argsort()[-5:][::-1]
        
        print(f"\n🧠 TOPIC {topic_idx}")
        print("-" * 20)
        
        for i, idx in enumerate(top_indices):
            # Get the text and the score
            sentence = df_text.iloc[idx]['text_segment']
            score = scores[idx]
            scope = df_text.iloc[idx]['scope']
            print(f"{i+1}. [{scope}] ({score:.2f}) \"{sentence}\"")
            
    print("\n" + "="*60)
    logger.info("Done. Use these descriptions to label your features in your Thesis.")

if __name__ == "__main__":
    interpret_topics()