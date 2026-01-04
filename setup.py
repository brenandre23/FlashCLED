from setuptools import setup, find_packages

setup(
    name='car_cewp',
    version='0.4.0',  # Bump version for major dependency updates
    description='Conflict Early Warning Pipeline for Central African Republic',
    author='B',
    packages=find_packages(),
    python_requires='>=3.10,<3.12',  # Updated to reflect modern environment
    
    install_requires=[
        # ============================
        # Core Data Science & ML
        # ============================
        'pandas>=1.5.3',
        'numpy>=1.24.3',
        'scipy>=1.10.1',
        'scikit-learn>=1.2.2',
        'statsmodels>=0.14.0',  # Added (used in analysis)
        'xgboost>=1.7.5',
        'lightgbm>=3.3.5',
        'joblib>=1.2.0',
        
        # ============================
        # Geospatial
        # ============================
        'geopandas>=0.13.2',
        'h3>=3.7.6',
        'shapely>=2.0.1',
        'pyproj>=3.5.0',
        'rtree>=1.0.1',
        'rasterio>=1.3.7',
        'affine>=2.4.0',
        'osmnx>=1.3.0',
        
        # ============================
        # Database
        # ============================
        'sqlalchemy>=2.0.15',   # Bumped to 2.0+ (Critical for code compatibility)
        'psycopg2-binary>=2.9.6',
        'geoalchemy2>=0.13.3',  # Added (Required for PostGIS uploads)
        'pyarrow>=12.0.0',
        'fastparquet>=2023.4.0',
        
        # ============================
        # Configuration
        # ============================
        'python-dotenv>=1.0.0',
        'pyyaml>=6.0',
        'click>=8.1.3',
        
        # ============================
        # Cloud & APIs
        # ============================
        'earthengine-api>=0.1.350',
        'google-cloud-bigquery>=3.11.0',
        'google-cloud-storage>=2.9.0',
        'google-auth>=2.0.0',
        'db-dtypes>=1.1.1',
        'yfinance>=0.2.18',
        'requests>=2.31.0',
        'cdsapi>=0.6.1',        # Added (For Copernicus/ERA5)
        
        # ============================
        # Utilities
        # ============================
        'tqdm>=4.65.0',
        'tenacity>=8.2.2',
    ],
    
    extras_require={
        'dev': [
            'pytest>=7.3.0',
            'pytest-cov>=4.0.0',
            'black>=23.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
            'ipython>=8.0.0',
            'jupyter>=1.0.0',
        ],
        'viz': [
            'matplotlib>=3.7.1',
            'seaborn>=0.12.2',
            'folium>=0.14.0',
            'branca>=0.6.0',
        ],
        'iom': [
            'openpyxl>=3.1.2',  # Often needed for IOM Excel files
        ],
    },
    
    entry_points={
        'console_scripts': [
            'cewp-run=main:CEWPPipeline',
        ]
    },
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: GIS',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)