"""
fetch_crisiswatch.py
====================
Purpose: 
  1. Scrape monthly intel from CrisisWatch (2003-Present).
  2. Extract 'Latent Topics' using ConfliBERT (Semantic Understanding).
  3. Geoparse locations using NLP + Nominatim -> H3 Grid.
  4. Output: 'features_crisiswatch' table (h3_index, date, topic_id, topic_name).

Methodology:
  - Text: Unstructured monthly summaries.
  - Spatial: Named Entity Recognition (SpaCy) -> Geocoding (OSM) -> H3 Index.
  - Temporal: Broadcasts Monthly data to the Daily/Weekly spine.
"""

import sys
import time
import json
import requests
import pandas as pd
import numpy as np
import spacy
import h3.api.basic_int as h3
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# --- Import Centralized Utilities ---
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, load_configs, upload_to_postgis, PATHS

# --- Configuration ---
SCHEMA = "car_cewp"
TABLE_NAME = "features_crisiswatch"
CACHE_FILE = PATHS['data_raw'] / "crisiswatch_geocache.json"
BASE_URL = "https://www.crisisgroup.org/crisiswatch/database"
COUNTRY_ID = "36"  # Central African Republic ID on their backend

# Rate limiter setup (OSM Policy: Max 1 request/sec)
geolocator = Nominatim(user_agent="car_conflict_thesis_research")
geocode_service = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)

class CrisisWatchPipeline:
    def __init__(self, resolution=5):
        self.resolution = resolution
        self.nlp = spacy.load("en_core_web_sm")
        self.geo_cache = self._load_cache()

    def _load_cache(self):
        if CACHE_FILE.exists():
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(CACHE_FILE, 'w') as f:
            json.dump(self.geo_cache, f)

    def scrape_data(self):
        """Iterates through paginated results."""
        logger.info("--- Phase 1: Scraping CrisisWatch ---")
        all_data = []
        page = 0
        
        while True:
            url = f"{BASE_URL}?location%5B%5D={COUNTRY_ID}&page={page}"
            try:
                r = requests.get(url, timeout=15)
                if r.status_code != 200:
                    break
                
                soup = BeautifulSoup(r.content, "html.parser")
                entries = soup.find_all("div", class_="crisiswatch-entry")
                
                if not entries:
                    logger.info(f"Page {page}: No entries found. Stopping.")
                    break
                
                logger.info(f"Page {page}: Found {len(entries)} briefs.")
                
                for entry in entries:
                    # Date extraction
                    date_div = entry.find("div", class_="crisiswatch-entry__date")
                    if not date_div: continue
                    
                    # Text extraction
                    text_div = entry.find("div", class_="crisiswatch-entry__text")
                    if not text_div: continue
                    
                    # Parse Date (Format usually "January 2023")
                    raw_date = date_div.text.strip()
                    try:
                        dt = pd.to_datetime(raw_date).date()
                    except:
                        continue # Skip invalid dates
                        
                    all_data.append({
                        "date": dt,
                        "text": text_div.text.strip()
                    })
                
                page += 1
                time.sleep(1) # Be polite to their server
                
            except Exception as e:
                logger.error(f"Scraping failed on page {page}: {e}")
                break
                
        return pd.DataFrame(all_data)

    def resolve_location_to_h3(self, loc_name):
        """Resolves string -> Lat/Lon -> H3 with Caching."""
        # normalization
        loc_clean = loc_name.lower().strip()
        
        # 1. Check Cache
        if loc_clean in self.geo_cache:
            lat, lon = self.geo_cache[loc_clean]
            if lat is None: return None
            return h3.geo_to_h3(lat, lon, self.resolution)
            
        # 2. Query Nominatim (with bias towards CAR)
        try:
            # Append country to reduce ambiguity
            query = f"{loc_name}, Central African Republic"
            loc = geocode_service(query)
            
            if loc:
                self.geo_cache[loc_clean] = (loc.latitude, loc.longitude)
                return h3.geo_to_h3(loc.latitude, loc.longitude, self.resolution)
            else:
                # Store None so we don't retry failed lookups
                self.geo_cache[loc_clean] = (None, None)
                return None
        except Exception as e:
            logger.warning(f"Geocoding error for {loc_name}: {e}")
            return None

    def run_pipeline(self):
        # 1. Scrape
        df = self.scrape_data()
        if df.empty:
            logger.warning("No data scraped.")
            return

        # 2. Topic Modeling (ConfliBERT)
        logger.info("--- Phase 2: Topic Modeling (ConfliBERT) ---")
        embedding_model = SentenceTransformer('snowood1/ConfliBERT-scr-uncased')
        topic_model = BERTopic(embedding_model=embedding_model, verbose=True)
        
        docs = df['text'].tolist()
        topics, probs = topic_model.fit_transform(docs)
        
        df['topic_id'] = topics
        # Map ID to human-readable label
        topic_info = topic_model.get_topic_info().set_index('Topic')['Name'].to_dict()
        df['topic_name'] = df['topic_id'].map(topic_info)

        # 3. Geoparsing
        logger.info("--- Phase 3: Geoparsing & H3 Mapping ---")
        
        # Blacklist generic terms that map to the whole country or random places
        BLACKLIST = {'central african republic', 'car', 'bangui', 'north', 'south', 'east', 'west'}
        
        final_rows = []
        
        for idx, row in df.iterrows():
            doc = self.nlp(row['text'])
            
            # Extract GPE/LOC
            locs = {ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC']}
            
            # Filter and Resolve
            valid_h3s = set()
            for loc in locs:
                if loc.lower() in BLACKLIST: continue
                
                h3_idx = self.resolve_location_to_h3(loc)
                if h3_idx:
                    valid_h3s.add(h3_idx)
            
            # If no location found, apply Country-Wide or Capital Fallback?
            # THESIS DECISION: Do NOT apply fallback. 
            # If text has no specific location, it contributes to "General Risk" 
            # but we can't map it to a specific cell without leakage.
            
            for h in valid_h3s:
                final_rows.append({
                    "date": row['date'],
                    "h3_index": h,
                    "cw_topic_id": row['topic_id'],
                    "cw_topic_name": row['topic_name'],
                    "cw_text_snippet": row['text'][:50] # For debug/audit
                })
            
            # Periodically save cache
            if idx % 10 == 0: self._save_cache()

        self._save_cache() # Final save
        
        return pd.DataFrame(final_rows)

def run(configs, engine):
    logger.info("STARTING CRISISWATCH INTEL PIPELINE")
    
    pipeline = CrisisWatchPipeline(resolution=configs['features']['spatial']['h3_resolution'])
    df_features = pipeline.run_pipeline()
    
    if df_features is None or df_features.empty:
        logger.warning("CrisisWatch pipeline produced no spatial data.")
        return

    # Ensure Types
    df_features['h3_index'] = df_features['h3_index'].astype('int64')
    df_features['date'] = pd.to_datetime(df_features['date'])
    
    # Upload
    logger.info(f"Uploading {len(df_features)} semantic features to {SCHEMA}.{TABLE_NAME}...")
    upload_to_postgis(
        engine, 
        df_features, 
        TABLE_NAME, 
        SCHEMA, 
        primary_keys=['h3_index', 'date', 'cw_topic_id'] # Composite PK allows multiple topics per cell/date
    )
    logger.info("CrisisWatch Ingestion Complete.")

if __name__ == "__main__":
    cfg = load_configs()
    if isinstance(cfg, tuple): cfg = {'data':cfg[0], 'features':cfg[1]}
    eng = get_db_engine()
    run(cfg, eng)