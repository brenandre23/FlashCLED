from setuptools import setup, find_packages

setup(
    name='car_cewp',
    version='0.2.2', # Bumped version
    description='Conflict Early Warning and Prediction system for Central African Republic',
    author='B',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        # --- Core Data Science ---
        'pandas>=1.3.0',
        'geopandas>=0.10.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        
        # --- Modeling ---
        'xgboost>=1.5.0',
        'joblib>=1.1.0',  # <--- ADDED (Required for train_models.py)
        
        # --- Database & Config ---
        'sqlalchemy>=1.4.0',
        'psycopg2-binary>=2.9.0',
        'python-dotenv>=0.19.0',
        'pyyaml>=5.4.0',
        'pyarrow>=6.0.0', # <--- ADDED (Required for Parquet caching)
        
        # --- Geospatial ---
        'h3>=3.7.0',
        'rasterio>=1.2.0',
        'osmnx>=1.1.0',
        'shapely>=1.8.0',
        'pyproj>=3.2.0',
        'rtree>=0.9.7',
        
        # --- Cloud & APIs ---
        'earthengine-api>=0.1.300',
        'google-cloud-bigquery>=3.0.0',
        'db-dtypes>=1.0.0',
        'yfinance>=0.2.0',
        
        # --- Utilities ---
        'tqdm>=4.60.0',
        'tenacity>=8.0.0',
        'requests>=2.26.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'black>=21.0',
            'flake8>=3.9.0',
            'ipython>=7.0.0',
            'jupyter>=1.0.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'car-cewp-pipeline=main:run_pipeline',
        ]
    },
)