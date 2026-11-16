"""Test setup: make the repo importable and stub heavy optional dependencies.

The leakage tests exercise pure pandas/numpy logic. Importing the pipeline
packages pulls in every ingestion/NLP module, so libraries that are irrelevant
here (and slow or unavailable in CI) are replaced by stubs when missing.
"""
import importlib
import sys
from pathlib import Path
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_OPTIONAL = [
    "torch", "sentence_transformers", "yfinance", "rasterio", "rasterio.mask", "rasterio.features",
    "rasterio.warp", "rasterio.merge", "rasterio.transform", "rasterio.windows", "rasterio.enums",
    "rasterstats", "osmnx", "google.cloud", "google.cloud.bigquery", "google.cloud.storage",
    "cdsapi", "tqdm", "tqdm.auto", "affine", "rapidfuzz", "networkx", "spacy", "bertopic",
    "umap", "hdbscan", "gensim", "geopy", "geopy.geocoders", "fiona", "pyogrio", "shap",
    "matplotlib", "matplotlib.pyplot", "seaborn", "statsmodels", "statsmodels.api",
    "transformers", "rtree", "pyproj", "openpyxl", "ee",
]


class _Meta(type):
    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return cls


class _Dummy(metaclass=_Meta):
    """Stands in for any class/function of a stubbed library. A real class, so
    isinstance/issubclass probes by numpy/scipy/sklearn keep working."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return self


class _StubModule(types.ModuleType):
    __path__ = []  # lets "import pkg.sub" resolve to registered stubs

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Dummy


for name in _OPTIONAL:
    try:
        importlib.import_module(name)
    except Exception:
        sys.modules[name] = _StubModule(name)
