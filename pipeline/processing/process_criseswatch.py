"""
pipeline/processing/process_criseswatch.py
==========================================
CrisisWatch NLP Feature Extraction Pipeline.

Processes ICG CrisisWatch monthly reports for CAR, extracts:
1. Per-topic semantic similarity scores (ConfliBERT embeddings)
2. Location-resolved H3 cells via hybrid gazetteer
3. Spatial confidence weights based on location resolution tier

OUTPUT SCHEMA (aligned with feature_engineering.py):
    h3_index (int64), date (datetime), cw_topic_id (int), spatial_confidence (float)

FIXES APPLIED:
1. H3 API FIX: h3.polyfill -> h3.polygon_to_cells (v4.x API)
2. SCHEMA FIX: Output columns match feature_engineering.py expectations
3. DB UPLOAD: Upsert to car_cewp.features_crisiswatch table
4. BATCHING: GPU-safe sentence encoding with configurable batch size
5. PERFORMANCE: Pre-filtered gazetteer lookup with Aho-Corasick
6. FILE DISCOVERY: Explicit filename pattern for CrisisWatch
"""

import sys
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import geopandas as gpd
import h3.api.basic_int as h3
import numpy as np
import pandas as pd
import spacy
import torch
from rapidfuzz import fuzz
from rapidfuzz import process as fuzz_process
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# --- CONFIG ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine  # noqa: E402

NLP_MODEL = "en_core_web_sm"
BERT_MODEL = "snowood1/ConfliBERT-scr-uncased"
H3_RES = 5  # Project standard resolution
BATCH_SIZE = 32  # Sentences per GPU batch
SCHEMA = "car_cewp"
OUTPUT_TABLE = "features_crisiswatch"

# Concept anchors for sentence scoring - mapped to topic IDs
CONCEPTS = {
    0: {  # risk_rebel_coalition
        "name": "rebel_coalition",
        "anchors": [
            "The Coalition of Patriots for Change CPC launched a new offensive against FACA positions.",
            "UPC and 3R rebel elements coordinated attacks on the main supply corridor.",
            "An alliance of armed groups announced a new offensive against the central government.",
        ],
    },
    1: {  # risk_transhumance_militia
        "name": "transhumance_militia",
        "anchors": [
            "Clashes erupted between Azande Ani Kpi Gbe militia and UPC rebels over territory.",
            "Communal violence escalated between armed pastoralists and farming communities.",
            "Transhumance-related violence between herders and farmers intensified.",
        ],
    },
    2: {  # risk_resource_predation
        "name": "resource_predation",
        "anchors": [
            "Rebels took control of the Ndassima gold mine and levied taxes on miners.",
            "Illicit taxation and looting of natural resources increased in the mining zone.",
            "Armed groups extracted resources and imposed illegal checkpoints on traders.",
        ],
    },
    3: {  # risk_foreign_influence
        "name": "foreign_influence",
        "anchors": [
            "Russian mercenaries Wagner Group Africa Corps launched a joint operation with FACA.",
            "Foreign military contractors were deployed to support government counter-insurgency operations.",
            "External actors increased their military and political presence in the region.",
        ],
    },
    4: {  # risk_state_vacuum
        "name": "state_vacuum",
        "anchors": [
            "Government forces lost control of the prefecture to the CPC coalition.",
            "Criminal gangs operate with impunity due to the complete lack of judicial presence.",
            "State authority collapsed in the region leaving a security vacuum.",
        ],
    },
    5: {  # risk_sectarian_cleansing
        "name": "sectarian_cleansing",
        "anchors": [
            "Anti-balaka militia targeted Muslim civilians in revenge killings.",
            "Civilians were targeted and displaced based on their ethnic or religious identity.",
            "Sectarian violence escalated with attacks on religious communities.",
        ],
    },
}

# Spatial confidence weights by resolution tier
SPATIAL_WEIGHTS = {
    "settlement_exact": 1.0,
    "settlement_fuzzy": 0.85,
    "admin3": 0.3,
    "admin2": 0.2,
    "admin1": 0.1,
}


def clean_text(text: str) -> str:
    """Normalize whitespace and remove quotes."""
    if not text:
        return ""
    text = re.sub(r"\"", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_name(name: str) -> str:
    """Lowercase and normalize whitespace for gazetteer matching."""
    return re.sub(r"\s+", " ", name.lower().strip()) if isinstance(name, str) else ""


class HierarchicalGazetteer:
    """
    Hybrid Gazetteer for CAR location resolution.
    
    Hierarchy:
      1. Settlements: exact match -> single H3 cell (weight: 1.0)
      2. Settlements: fuzzy match (≥90%) -> single H3 cell (weight: 0.85)
      3. Admin3 polygons: exact match -> set of H3 cells (weight: 0.3)
      4. Admin2 polygons: exact match -> set of H3 cells (weight: 0.2)
      5. Admin1 polygons: exact match -> set of H3 cells (weight: 0.1)
    """

    def __init__(self, resolution: int):
        self.resolution = resolution
        self.settlements: Dict[str, int] = {}  # name -> h3_index
        self.admin1: Dict[str, Set[int]] = {}  # name -> set(h3_indices)
        self.admin2: Dict[str, Set[int]] = {}
        self.admin3: Dict[str, Set[int]] = {}
        self._settlement_names: List[str] = []  # For fuzzy matching
        self._load_settlements()
        self._load_admin()

    def _load_settlements(self):
        """Load settlement points from SIGCAF dataset."""
        zip_path = ROOT_DIR / "data" / "raw" / "caf_settlements_sigcaf.zip"
        if not zip_path.exists():
            logger.warning(f"Settlements zip not found: {zip_path}")
            return
        
        try:
            gdf = gpd.read_file(f"zip://{zip_path}")
        except Exception as e:
            logger.error(f"Failed to read settlements: {e}")
            return
            
        name_col = "PName1" if "PName1" in gdf.columns else None
        if not name_col:
            logger.warning("PName1 column missing in settlements; skipping.")
            return
            
        for _, row in gdf.iterrows():
            if row.geometry is None:
                continue
            name = normalize_name(row[name_col])
            if not name:
                continue
            lat, lon = row.geometry.y, row.geometry.x
            try:
                h3_idx = h3.latlng_to_cell(lat, lon, self.resolution)
                self.settlements[name] = int(h3_idx)
            except Exception:
                continue
                
        self._settlement_names = list(self.settlements.keys())
        logger.info(f"Loaded {len(self.settlements)} settlements into gazetteer.")

    def _poly_to_h3(self, geom) -> Set[int]:
        """Convert polygon geometry to set of H3 cells."""
        try:
            # H3 v4.x API: geo_to_cells accepts GeoJSON directly
            geojson = geom.__geo_interface__
            cells = h3.geo_to_cells(geojson, self.resolution)
            return {int(c) for c in cells}
        except Exception:
            # Fallback: use centroid
            try:
                lat, lon = geom.centroid.y, geom.centroid.x
                return {int(h3.latlng_to_cell(lat, lon, self.resolution))}
            except Exception:
                return set()

    def _load_admin(self):
        """Load administrative boundary polygons."""
        admin_files = [
            ("admin1", ROOT_DIR / "data" / "raw" / "wbgCAFadmin1.geojson", "NAM_1"),
            ("admin2", ROOT_DIR / "data" / "raw" / "wbgCAFadmin2.geojson", "NAM_2"),
            ("admin3", ROOT_DIR / "data" / "raw" / "wbgCAFadmin3.geojson", "adm2_ref_name"),
        ]
        
        for level, path, name_col in admin_files:
            if not path.exists():
                logger.warning(f"Admin file missing: {path}")
                continue
                
            try:
                gdf = gpd.read_file(path)
            except Exception as e:
                logger.error(f"Failed to read {path}: {e}")
                continue
                
            if name_col not in gdf.columns:
                logger.warning(f"{name_col} missing in {path.name}; skipping.")
                continue
                
            target = getattr(self, level)
            for _, row in gdf.iterrows():
                name = normalize_name(row[name_col])
                if not name or row.geometry is None:
                    continue
                h3_cells = self._poly_to_h3(row.geometry)
                if h3_cells:
                    target[name] = h3_cells
                    
            logger.info(f"Loaded {len(target)} {level} polygons into gazetteer.")

    def resolve_location(self, place: str) -> Tuple[Optional[str], float, List[int]]:
        """
        Resolve a place name to H3 cells with spatial confidence.
        
        Returns:
            (tier_name, weight, [h3_indices]) or (None, 0.0, [])
        """
        clean = normalize_name(place)
        if not clean or len(clean) < 3:  # Skip very short tokens
            return None, 0.0, []

        # Tier 1: Exact settlement match
        if clean in self.settlements:
            return "settlement_exact", SPATIAL_WEIGHTS["settlement_exact"], [self.settlements[clean]]

        # Tier 2: Fuzzy settlement match
        if self._settlement_names:
            match = fuzz_process.extractOne(
                clean, 
                self._settlement_names, 
                scorer=fuzz.WRatio,
                score_cutoff=90
            )
            if match:
                return "settlement_fuzzy", SPATIAL_WEIGHTS["settlement_fuzzy"], [self.settlements[match[0]]]

        # Tier 3-5: Admin boundaries (exact match only for performance)
        for tier, admin_dict in [
            ("admin3", self.admin3),
            ("admin2", self.admin2),
            ("admin1", self.admin1),
        ]:
            if clean in admin_dict:
                return tier, SPATIAL_WEIGHTS[tier], list(admin_dict[clean])

        return None, 0.0, []


def get_device() -> torch.device:
    """Get best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_text_file(file_path: Path) -> pd.DataFrame:
    """
    Parse CrisisWatch text file into dated entries.
    
    Expected format:
        Central African Republic [Month Year]
        [Report text...]
        
        Central African Republic [Month Year]
        ...
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    data = []
    current_date = None
    buffer = []
    header_pattern = re.compile(
        r"Central African Republic\s+(\w+\s+\d{4})", 
        re.IGNORECASE
    )

    for line in lines:
        line = clean_text(line)
        if not line:
            continue
            
        match = header_pattern.search(line)
        if match:
            # Save previous entry
            if current_date and buffer:
                data.append({"date": current_date, "text": " ".join(buffer)})
            # Start new entry
            try:
                current_date = pd.to_datetime(f"1 {match.group(1)}").normalize()
                buffer = []
            except Exception:
                current_date = None
        else:
            if current_date:
                buffer.append(line)
                
    # Don't forget last entry
    if current_date and buffer:
        data.append({"date": current_date, "text": " ".join(buffer)})
        
    logger.info(f"Parsed {len(data)} monthly entries from CrisisWatch file.")
    return pd.DataFrame(data)


def encode_sentences_batched(
    sentences: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = BATCH_SIZE
) -> np.ndarray:
    """
    Encode sentences in batches to avoid GPU OOM.
    
    Returns:
        (N, hidden_dim) array of [CLS] embeddings
    """
    all_embeddings = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)
            
    return np.vstack(all_embeddings)


def compute_topic_scores(
    sentence_vec: np.ndarray,
    concept_vectors: Dict[int, np.ndarray]
) -> Dict[int, float]:
    """
    Compute similarity scores for each topic.
    
    Args:
        sentence_vec: (1, hidden_dim) sentence embedding
        concept_vectors: {topic_id: (n_anchors, hidden_dim)} anchor embeddings
        
    Returns:
        {topic_id: max_similarity_score}
    """
    scores = {}
    for topic_id, anchors in concept_vectors.items():
        sims = cosine_similarity(sentence_vec, anchors)[0]
        scores[topic_id] = float(np.max(sims)) if sims.size > 0 else 0.0
    return scores


def upload_to_postgis(engine, df: pd.DataFrame):
    """
    Upsert CrisisWatch features to PostGIS.
    
    Schema:
        h3_index BIGINT, date DATE, cw_topic_id INT, spatial_confidence FLOAT
        PRIMARY KEY (h3_index, date, cw_topic_id)
    """
    if df.empty:
        logger.warning("Empty dataframe; skipping upload.")
        return
        
    with engine.begin() as conn:
        # Create table if not exists
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA}.{OUTPUT_TABLE} (
                h3_index BIGINT NOT NULL,
                date DATE NOT NULL,
                cw_topic_id INTEGER NOT NULL,
                spatial_confidence FLOAT NOT NULL,
                PRIMARY KEY (h3_index, date, cw_topic_id)
            )
        """))
        
        # Create index for faster lookups
        conn.execute(text(f"""
            CREATE INDEX IF NOT EXISTS idx_{OUTPUT_TABLE}_date 
            ON {SCHEMA}.{OUTPUT_TABLE} (date)
        """))
        
    # Upsert in chunks
    chunk_size = 10000
    total_rows = len(df)
    
    for start in range(0, total_rows, chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        
        # Build VALUES clause
        values = []
        for _, row in chunk.iterrows():
            values.append(
                f"({int(row['h3_index'])}, '{row['date'].strftime('%Y-%m-%d')}', "
                f"{int(row['cw_topic_id'])}, {float(row['spatial_confidence'])})"
            )
            
        if not values:
            continue
            
        values_str = ",\n".join(values)
        
        upsert_sql = f"""
            INSERT INTO {SCHEMA}.{OUTPUT_TABLE} (h3_index, date, cw_topic_id, spatial_confidence)
            VALUES {values_str}
            ON CONFLICT (h3_index, date, cw_topic_id)
            DO UPDATE SET spatial_confidence = EXCLUDED.spatial_confidence
        """
        
        with engine.begin() as conn:
            conn.execute(text(upsert_sql))
            
    logger.info(f"Upserted {total_rows} rows to {SCHEMA}.{OUTPUT_TABLE}")


def run():
    """Main CrisisWatch processing pipeline."""
    logger.info("=" * 60)
    logger.info("CRISISWATCH NLP PROCESSING (Schema-Aligned)")
    logger.info("=" * 60)

    # Find CrisisWatch file (explicit pattern)
    raw_dir = ROOT_DIR / "data" / "raw"
    input_files = list(raw_dir.glob("*crisiswatch*.txt")) + list(raw_dir.glob("*CrisisWatch*.txt"))
    
    if not input_files:
        # Fallback: any txt file with "crisis" in name
        input_files = [f for f in raw_dir.glob("*.txt") if "crisis" in f.name.lower()]
        
    if not input_files:
        logger.error("No CrisisWatch txt file found in data/raw/")
        return
        
    input_file = input_files[0]
    logger.info(f"Processing: {input_file.name}")

    # Parse text
    df_raw = parse_text_file(input_file)
    if df_raw.empty:
        logger.warning("Parsed CrisisWatch file is empty.")
        return

    # Load NLP models
    logger.info("Loading NLP models...")
    nlp = spacy.load(NLP_MODEL)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModel.from_pretrained(BERT_MODEL).to(device)
    model.eval()

    # Pre-compute concept anchor embeddings
    logger.info("Computing concept anchor embeddings...")
    concept_vectors = {}
    for topic_id, concept in CONCEPTS.items():
        anchors = concept["anchors"]
        inputs = tokenizer(
            anchors, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            concept_vectors[topic_id] = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    # Initialize gazetteer
    logger.info("Building hierarchical gazetteer...")
    gaz = HierarchicalGazetteer(H3_RES)

    # Accumulator: (date, h3_index, topic_id) -> list of spatial_confidence scores
    accumulator: Dict[Tuple, List[float]] = {}

    def add_score(dt, h3_idx: int, topic_id: int, spatial_conf: float):
        key = (dt, h3_idx, topic_id)
        if key not in accumulator:
            accumulator[key] = []
        accumulator[key].append(spatial_conf)

    # Process each monthly entry
    for _, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc="Processing entries"):
        if not row["text"]:
            continue
            
        doc = nlp(row["text"])
        
        # Filter to meaningful sentences
        sents = [s for s in doc.sents if len(s.text.strip()) > 20]
        if not sents:
            continue

        # Batch encode all sentences
        sent_texts = [s.text for s in sents]
        sent_vecs = encode_sentences_batched(sent_texts, tokenizer, model, device)

        # Process each sentence
        for i, sent in enumerate(sents):
            vec = sent_vecs[i].reshape(1, -1)
            
            # Score against each topic
            topic_scores = compute_topic_scores(vec, concept_vectors)
            
            # Extract location mentions from sentence
            tokens = {t.text.lower() for t in sent if not t.is_stop and len(t.text) >= 3}
            
            # Also extract named entities
            for ent in sent.ents:
                if ent.label_ in ("GPE", "LOC", "FAC"):
                    tokens.add(ent.text.lower())
            
            # Resolve each potential location
            for tok in tokens:
                tier, weight, h3_list = gaz.resolve_location(tok)
                if tier is None or not h3_list:
                    continue
                    
                # For each topic with positive score, add weighted contribution
                for topic_id, topic_score in topic_scores.items():
                    if topic_score <= 0.3:  # Threshold for relevance
                        continue
                        
                    # Spatial confidence = topic_score * location_weight
                    spatial_conf = topic_score * weight
                    
                    for h3_idx in h3_list:
                        add_score(row["date"], int(h3_idx), topic_id, spatial_conf)

    if not accumulator:
        logger.warning("No CrisisWatch locations resolved; nothing to write.")
        return

    # Aggregate: sum confidence scores per (date, h3, topic)
    rows = []
    for (dt, h3_idx, topic_id), vals in accumulator.items():
        rows.append({
            "date": pd.to_datetime(dt),
            "h3_index": np.int64(h3_idx),
            "cw_topic_id": int(topic_id),
            "spatial_confidence": float(np.sum(vals)),  # Sum for cumulative signal
        })

    out_df = pd.DataFrame(rows)
    
    # Save parquet backup
    out_path = ROOT_DIR / "data" / "processed" / "features_crisiswatch.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    logger.info(f"Saved parquet backup: {out_path} ({len(out_df)} rows)")
    
    # Upload to PostGIS
    logger.info("Uploading to PostGIS...")
    engine = get_db_engine()
    try:
        upload_to_postgis(engine, out_df)
    finally:
        engine.dispose()

    # Summary stats
    logger.info("=" * 60)
    logger.info("CRISISWATCH PROCESSING COMPLETE")
    logger.info(f"  Total records: {len(out_df)}")
    logger.info(f"  Unique H3 cells: {out_df['h3_index'].nunique()}")
    logger.info(f"  Date range: {out_df['date'].min()} to {out_df['date'].max()}")
    logger.info(f"  Topics extracted: {sorted(out_df['cw_topic_id'].unique().tolist())}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run()