from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

RANDOM_SEED = 42
