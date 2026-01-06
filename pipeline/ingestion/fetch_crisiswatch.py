"""
fetch_crisiswatch.py
====================
Fetches CrisisWatch intel (Aug 2003 - present), applies topic modeling & geoparsing.
Note: CrisisWatch started Aug 2003; earlier dates in config will have no data.
"""

import sys
import json
import feedparser
import pandas as pd
import numpy as np
import spacy
import h3.api.basic_int as h3
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import date
from dateutil import parser as date_parser
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from utils import logger, get_db_engine, load_configs, upload_to_postgis, PATHS

# --- Constants ---
SCHEMA = "car_cewp"
TABLE_NAME = "features_crisiswatch"
CACHE_FILE = PATHS['data_raw'] / "crisiswatch_geocache.json"
CRISISWATCH_EARLIEST = date(2003, 8, 1)

# RSS Feeds
RSS_CAR = "https://www.crisisgroup.org/rss/5"
RSS_GLOBAL = "https://www.crisisgroup.org/rss/crisiswatch"

# Known CAR locations (lat, lon)
CAR_LOCATIONS = {
    'bangui': (4.3947, 18.5582), 'bimbo': (4.2567, 18.4153), 'bria': (6.5364, 21.9842),
    'bambari': (5.7631, 20.6672), 'berberati': (4.2614, 15.7892), 'bossangoa': (6.4928, 17.4553),
    'carnot': (4.9422, 15.8736), 'ndele': (8.4089, 20.6533), 'obo': (5.4000, 26.4917),
    'zemio': (5.0314, 25.1383), 'bouar': (5.9500, 15.6000), 'kaga-bandoro': (6.9864, 19.1831),
    'nola': (3.5250, 16.0472), 'sibut': (5.7178, 19.0736), 'bangassou': (4.7414, 22.8189),
    'paoua': (7.2333, 16.4333), 'batangafo': (7.3000, 18.2833), 'birao': (10.2833, 22.7833),
}

LOCATION_BLACKLIST = {
    'central african republic', 'car', 'republic', 'north', 'south', 'east', 'west',
    'central', 'africa', 'african', 'france', 'un', 'russia', 'china', 'eu'
}

geolocator = Nominatim(user_agent="car_conflict_thesis_v2")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)


class CrisisWatchPipeline:
    def __init__(self, resolution, start_date, end_date):
        self.resolution = resolution
        self.start_date = max(start_date, CRISISWATCH_EARLIEST)
        self.end_date = end_date
        self.nlp = spacy.load("en_core_web_sm")
        self.geo_cache = self._load_cache()
        self.geo_cache.update({k: v for k, v in CAR_LOCATIONS.items() if k not in self.geo_cache})

        if start_date < CRISISWATCH_EARLIEST:
            logger.warning(f"CrisisWatch data starts {CRISISWATCH_EARLIEST}; no data for {start_date} to {CRISISWATCH_EARLIEST}")

    def _load_cache(self):
        if CACHE_FILE.exists():
            return json.load(open(CACHE_FILE))
        return {}

    def _save_cache(self):
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        json.dump(self.geo_cache, open(CACHE_FILE, 'w'))

    def fetch_rss(self):
        """Fetch CAR entries from RSS feeds within date range."""
        logger.info(f"Fetching CrisisWatch RSS: {self.start_date} to {self.end_date}")
        
        for url, name in [(RSS_CAR, "CAR"), (RSS_GLOBAL, "Global")]:
            logger.info(f"  Trying {name} feed...")
            feed = feedparser.parse(url)
            
            entries = []
            for e in feed.entries:
                try:
                    pub_date = date_parser.parse(getattr(e, 'published', getattr(e, 'updated', ''))).date()
                except:
                    continue
                
                if not (self.start_date <= pub_date <= self.end_date):
                    continue
                
                text = BeautifulSoup(getattr(e, 'summary', '') or getattr(e, 'description', ''), 'html.parser').get_text(' ', strip=True)
                
                # Filter global feed for CAR content
                if url == RSS_GLOBAL:
                    combined = (getattr(e, 'title', '') + text).lower()
                    if not any(kw in combined for kw in ['central african republic', 'bangui', 'centrafrique']):
                        continue
                
                if len(text) > 50:
                    entries.append({'date': pub_date, 'text': text})
            
            if entries:
                logger.info(f"  Found {len(entries)} entries")
                return pd.DataFrame(entries).drop_duplicates(subset=['date', 'text'])
        
        logger.warning("No RSS data retrieved")
        return pd.DataFrame()

    def resolve_h3(self, loc_name):
        """Resolve location name to H3 index."""
        key = loc_name.lower().strip()
        if key in self.geo_cache:
            coords = self.geo_cache[key]
            return h3.geo_to_h3(coords[0], coords[1], self.resolution) if coords[0] else None
        
        try:
            loc = geocode(f"{loc_name}, Central African Republic")
            self.geo_cache[key] = (loc.latitude, loc.longitude) if loc else (None, None)
            return h3.geo_to_h3(loc.latitude, loc.longitude, self.resolution) if loc else None
        except:
            return None

    def run(self):
        df = self.fetch_rss()
        if df.empty:
            return None

        # Topic modeling
        logger.info("Running topic modeling...")
        try:
            model = BERTopic(embedding_model=SentenceTransformer('snowood1/ConfliBERT-scr-uncased'), 
                           verbose=False, min_topic_size=2)
            topics, _ = model.fit_transform(df['text'].tolist())
            df['topic_id'] = topics
            df['topic_name'] = df['topic_id'].map(model.get_topic_info().set_index('Topic')['Name'].to_dict())
        except Exception as e:
            logger.warning(f"Topic modeling failed: {e}")
            df['topic_id'], df['topic_name'] = 0, "general"

        # Geoparsing
        logger.info("Geoparsing locations...")
        rows = []
        for _, r in df.iterrows():
            locs = {ent.text for ent in self.nlp(r['text']).ents if ent.label_ in ['GPE', 'LOC']}
            locs.update(k.title() for k in CAR_LOCATIONS if k in r['text'].lower())
            
            h3_cells = {self.resolve_h3(loc) for loc in locs if loc.lower() not in LOCATION_BLACKLIST}
            h3_cells.discard(None)
            
            if not h3_cells:  # Default to Bangui
                h3_cells = {h3.geo_to_h3(4.3947, 18.5582, self.resolution)}
            
            for cell in h3_cells:
                rows.append({
                    'date': r['date'], 'h3_index': cell,
                    'cw_topic_id': r['topic_id'], 'cw_topic_name': r['topic_name'],
                    'cw_text_snippet': r['text'][:50]
                })
        
        self._save_cache()
        return pd.DataFrame(rows) if rows else None


def run(configs, engine):
    logger.info("STARTING CRISISWATCH PIPELINE")
    
    data_cfg = configs.get('data', configs[0] if isinstance(configs, tuple) else {})
    feat_cfg = configs.get('features', configs[1] if isinstance(configs, tuple) else {})
    
    start = pd.to_datetime(data_cfg['global_date_window']['start_date']).date()
    end = pd.to_datetime(data_cfg['global_date_window']['end_date']).date()
    resolution = feat_cfg['spatial']['h3_resolution']
    
    pipeline = CrisisWatchPipeline(resolution, start, end)
    df = pipeline.run()
    
    if df is None or df.empty:
        logger.warning("No CrisisWatch data produced")
        return
    
    df['h3_index'] = df['h3_index'].astype('int64')
    df['date'] = pd.to_datetime(df['date'])
    
    logger.info(f"Uploading {len(df)} records to {SCHEMA}.{TABLE_NAME}")
    upload_to_postgis(engine, df, TABLE_NAME, SCHEMA, primary_keys=['h3_index', 'date', 'cw_topic_id'])
    logger.info("CrisisWatch complete")


if __name__ == "__main__":
    cfg = load_configs()
    if isinstance(cfg, tuple):
        cfg = {'data': cfg[0], 'features': cfg[1]}
    run(cfg, get_db_engine())
