"""Scraper-specific configuration."""
import sys
from pathlib import Path
from datetime import date

# ~~ Data Directory Config ~~ # 

PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"

ABSTRACT_DIR = DATA_DIR / "abstracts"
PDF_DIR = DATA_DIR / "pdfs"
SOURCE_DIR = DATA_DIR / "source"
RAW_TEX_DIR = DATA_DIR / "raw_tex"
CLEAN_TEX_DIR = DATA_DIR / "clean_tex"
LOG_DIR = DATA_DIR / "corpus_build.log"

# ~~ Defaults for CLI Arguments ~~ #

CLI_DEFAULTS = {
    "start_date" : "1900-01-01",
    "end_date" : date.today().strftime("%Y-%m-%d"),
    "max_papers" : 1000,
    "verbose" : True,
}

# ~~ Defaults for LaTeX Cleaning ~~ #

SECTIONS_TO_REMOVE = [
    'acknowledgements',
    'acknowledgments',
    'references',
    'bibliography',
]

# ~~ API Reqeust Config ~~ #

REQUEST_CONFIG = {
    "timeout": 30,
    "delay_between_requests": 3,  # seconds
    "max_retries": 3,
    "max_page_size": 250,
}