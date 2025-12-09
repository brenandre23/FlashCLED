# init_db.py
"""
Pipeline: Initialization & Schema Enforcement
Task: 
  1. Sets up PostgreSQL extensions (PostGIS, H3).
  2. Creates the base schema.
  3. Enforces H3 BIGINT standards (Schema Repair).
"""
import sys
from pathlib import Path
from sqlalchemy import text, inspect

# --- Setup Project Root ---
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import centralized utils to ensure consistency
from utils import logger, get_db_engine

# Configuration
SCHEMA = "car_cewp"

def init_extensions(conn):
    """Enables required PostgreSQL extensions."""
    logger.info("Initializing extensions...")
    extensions = ["postgis", "h3", "h3_postgis"]
    
    for ext in extensions:
        conn.execute(text(f"CREATE EXTENSION IF NOT EXISTS {ext} CASCADE;"))
    logger.info(f"Extensions verified: {', '.join(extensions)}")

def init_schema(conn):
    """Creates the project schema."""
    logger.info(f"Verifying schema '{SCHEMA}'...")
    conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))

def enforce_h3_standards(conn):
    """
    CRITICAL FIX: Checks existing tables and migrates h3_index 
    from VARCHAR to BIGINT if necessary.
    """
    logger.info("Checking H3 data type consistency...")
    
    # Tables that historically had issues
    target_tables = ["acled_events", "features_dynamic_daily"]
    
    inspector = inspect(conn)
    
    for table in target_tables:
        # 1. Check if table exists in our schema
        if inspector.has_table(table, schema=SCHEMA):
            # 2. Get columns
            columns = {c['name']: c for c in inspector.get_columns(table, schema=SCHEMA)}
            
            if 'h3_index' in columns:
                col_type = str(columns['h3_index']['type']).upper()
                
                # Check for String/Varchar types
                if 'CHAR' in col_type or 'TEXT' in col_type:
                    logger.warning(f"⚠️  Fixing schema for {SCHEMA}.{table}: h3_index is {col_type}, converting to BIGINT...")
                    try:
                        # Perform the migration
                        # Note: We use x'...'::bigint if it's a hex string, or cast directly if numeric string
                        # Safer approach: Try H3-extension conversion or direct cast
                        conn.execute(text(f"""
                            ALTER TABLE {SCHEMA}.{table}
                            ALTER COLUMN h3_index TYPE BIGINT 
                            USING (
                                CASE 
                                    WHEN h3_index ~ '^[0-9]+$' THEN h3_index::bigint  -- Already numeric string
                                    ELSE ('x' || h3_index)::bit(64)::bigint           -- Hex string
                                END
                            );
                        """))
                        logger.info(f"   ✓ {table} converted successfully.")
                    except Exception as e:
                        logger.error(f"   ❌ Failed to convert {table}: {e}")
                else:
                    logger.info(f"   ✓ {table} is already compliant ({col_type}).")

def run():
    logger.info("--- STARTING DATABASE INITIALIZATION (Step 0) ---")
    engine = None
    try:
        # Use the robust engine from utils
        engine = get_db_engine()
        
        with engine.begin() as conn:
            # 1. Core Setup
            init_extensions(conn)
            init_schema(conn)
            
            # 2. Repair/Enforce Standards
            enforce_h3_standards(conn)

        logger.info("--- DATABASE INITIALIZATION COMPLETE ---")
        
    except Exception as e:
        logger.critical(f"Database initialization failed: {e}")
        sys.exit(1)
    finally:
        if engine:
            engine.dispose()

if __name__ == "__main__":
    run()