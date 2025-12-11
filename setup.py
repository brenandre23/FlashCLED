from setuptools import setup, find_packages

setup(
    name='car_cewp',
    version='0.3.1',  # Bump version for new model support
    description='Conflict Early Warning Pipeline for Central African Republic',
    author='B',
    packages=find_packages(),
    python_requires='>=3.9,<3.12',
    
    install_requires=[
        # ============================
        # Core Data Science & ML
        # ============================
        'pandas>=1.3.0,<3.0.0',
        'numpy>=1.21.0,<2.0.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'xgboost>=1.5.0',
        'lightgbm>=3.3.0',  # <--- Added LightGBM
        'joblib>=1.1.0',
        
        # ============================
        # Geospatial
        # ============================
        'geopandas>=0.10.0',
        'h3>=3.7.0,<5.0.0',
        'shapely>=1.8.0,<3.0.0',
        'pyproj>=3.2.0',
        'rtree>=0.9.7',
        'rasterio>=1.2.0',
        'osmnx>=1.1.0',
        
        # ============================
        # Database
        # ============================
        'sqlalchemy>=1.4.0,<2.0.0',
        'psycopg2-binary>=2.9.0',
        'pyarrow>=6.0.0',
        
        # ============================
        # Configuration
        # ============================
        'python-dotenv>=0.19.0',
        'pyyaml>=5.4.0',
        
        # ============================
        # Cloud & APIs
        # ============================
        'earthengine-api>=0.1.300',
        'google-cloud-bigquery>=3.0.0',
        'google-auth>=2.0.0',
        'db-dtypes>=1.0.0',
        'yfinance>=0.2.0',
        'requests>=2.26.0',
        
        # ============================
        # Utilities
        # ============================
        'tqdm>=4.60.0',
        'tenacity>=8.0.0',
    ],
    
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=3.0.0',
            'black>=21.0',
            'flake8>=3.9.0',
            'mypy>=0.900',
            'ipython>=7.0.0',
            'jupyter>=1.0.0',
        ],
        'viz': [
            'matplotlib>=3.4.0',
            'seaborn>=0.11.0',
            'folium>=0.12.0',
        ],
        'iom': [
            'dtmapi>=0.1.0',
        ],
    },
    
    entry_points={
        'console_scripts': [
            'cewp-run=main:CEWPPipeline',
        ]
    },
    
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: GIS',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)