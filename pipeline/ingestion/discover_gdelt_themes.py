"""
discover_gdelt_themes.py
========================
Scans GDELT GKG to find the "True Universe" of themes in CAR and writes the top 300 list.
"""
import sys
import pandas as pd
import requests
import io
import zipfile
import random
import multiprocessing
from collections import Counter
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Setup ---
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))
from utils import logger  # noqa: E402

# CONFIG
COUNTRY_FILTER = "CT"  # Central African Republic
SAMPLE_SIZE_FILES = 100  # Scan 100 random files (approx 10 days of data) to find the distribution
MAX_WORKERS = max(4, multiprocessing.cpu_count() - 1)
OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "gdelt_top_300_themes.csv"


def get_random_file_urls():
    """Fetch random GKG file URLs from the last 3 years."""
    print("Fetching Master File List...")
    try:
        r = requests.get("http://data.gdeltproject.org/gdeltv2/masterfilelist.txt")
        lines = r.text.split("\n")
    except Exception:
        return []

    # Filter for GKG zips from 2023-2025
    recent_lines = [
        l for l in lines if "gkg.csv.zip" in l and any(y in l for y in ["2023", "2024", "2025"])
    ]
    if not recent_lines:
        return []

    # Return random sample
    return [l.split(" ")[-1] for l in random.sample(recent_lines, min(len(recent_lines), SAMPLE_SIZE_FILES))]


def scan_file(url):
    """Download one file and count ALL themes for CAR."""
    local_counter = Counter()
    try:
        r = requests.get(url, timeout=15)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(
            z.open(z.namelist()[0]),
            sep="\t",
            header=None,
            encoding="latin1",
            usecols=[7, 9],
            names=["themes", "locations"],
        )

        # Filter for CAR
        df = df[df["locations"].str.contains(COUNTRY_FILTER, na=False)]

        # Count every theme
        for theme_str in df["themes"].dropna():
            themes = theme_str.split(";")
            local_counter.update([t for t in themes if t])  # unexpected empty strings

        return local_counter
    except Exception:
        return Counter()


def run():
    print(f"dYs? Starting Wide Net Scan on {SAMPLE_SIZE_FILES} files...")
    urls = get_random_file_urls()

    global_counter = Counter()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(scan_file, url): url for url in urls}
        for future in tqdm(as_completed(futures), total=len(urls), desc="Scanning Universe"):
            global_counter.update(future.result())

    # Save Results
    print(f"\nƒo. Scan Complete. Found {len(global_counter)} unique themes.")

    # Convert to DataFrame
    df_themes = pd.DataFrame(global_counter.most_common(), columns=["Theme", "Count"])

    # Save the "Top 300" (The Signals)
    top_300 = df_themes.head(300)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    top_300.to_csv(OUTPUT_FILE, index=False)

    print(f"dY'_ Saved Top 300 themes to: {OUTPUT_FILE}")
    print("\nTOP 10 THEMES FOUND:")
    print(top_300.head(10).to_markdown(index=False))


def main():
    run()


if __name__ == "__main__":
    main()
