"""Main entry point for LHCb paper scraper."""

import sys
from pathlib import Path

if __name__ == "__main__":
    scraper_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(scraper_dir))

import settings
from scraper.builder.build_corpus import main as build_corpus_main

if __name__ == "__main__":
    build_corpus_main()
