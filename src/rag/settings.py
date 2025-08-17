"""Scraper-specific configuration."""
import sys
from pathlib import Path

# ~~ Global Config ~~ # 

PROJECT_ROOT = Path(__file__).parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"

EMBEDDING_CONFIG = {
    "model": "BAAI/bge-small-en-v1.5",
    "device": "cuda",  # or "cpu"
    "overwrite_existing": False,
}

# Input directories from the scraper
ABSTRACT_DIR = DATA_DIR / "abstracts"
CLEAN_TEX_DIR = DATA_DIR / "clean_tex"

MAX_WORKERS = 30

LATEX_ELEMENT_TYPES = ["figure", "table", "sidewaystable"]

# ~~ Tree RAG Config ~~ #

TREE_RAG_CACHE = DATA_DIR / "tree_rag_cache"

PAPER_TREE_CONFIG = {
    "max_chunks" : 8,
    "min_section_length" : 50,
    "section_max_tokens" : 500,
    "overwrite_existing" : False,
}

SUMMARY_CONFIG = {
    "request_delay" : 10,
    "batch_size" : 200,
    "max_threads" : 200,
    "model" : "gpt-5-nano",
    "overwrite_existing" : False,

}

# Output directories for cache of RAG databases
GRAPH_RAG_CACHE = DATA_DIR / "graph_rag_cache"

# ~~ Graph RAG Config ~~ #

GRAPH_RAG_CACHE = DATA_DIR / "graph_rag_cache"

GRAPH_CONFIG = {
    "model" : "gpt-5-mini",
    "section_ignore_keywords" : ["introduction", "detector", "dataset", "appendi", "supplement", "end matter", "acknowle", "aknowle"],
    "overwrite_existing" : False,
    "max_threads" : 10,
}

CANONICALIZE_CONFIG = {
    "stop_ratio": 0.2,
    "max_cluster_size": 50,
    "max_threads" : 100,
    "model": "gpt-5-nano",
    "overwrite_existing": False,
    "type_instructions": {
        "uncertainty_source": (
            "Merge only when they describe the same systematic effect with essentially the same physical origin and mechanism. "
            "Guidelines for merging:\n"
            "• MERGE: True synonyms or equivalent labels for the same effect (e.g., 'tracking efficiency' ≈ 'track reconstruction efficiency')\n"
            "• MERGE: Same systematic effect described at different levels of detail (e.g., general 'trigger efficiency' with specific 'muon trigger efficiency')\n"
            "• MERGE: Same physical source with different technical implementations (e.g., 'J/ψ calibration' and 'tag-and-probe calibration' for tracking)\n"
            "• DO NOT MERGE: Different detector subsystems (tracking vs PID vs trigger vs calorimeter)\n"
            "• DO NOT MERGE: Different analysis stages (selection vs fitting vs background modeling)\n"
            "• DO NOT MERGE: Different particle types unless they're the same systematic (e.g., 'muon efficiency' ≠ 'pion efficiency')\n"
            "• DO NOT MERGE: Different theoretical/modeling choices (lineshape models vs background models vs production models)\n"
            "• BORDERLINE: Use physics judgment - if two sources would be correlated in a global analysis, consider merging"
        ),
        "method": (
            "Merge when they describe the same evaluation technique or procedural approach, even if applied to different uncertainties. "
            "Guidelines for merging:\n"
            "• MERGE: Same fundamental technique with different applications (e.g., 'tag-and-probe for muons' and 'tag-and-probe for kaons')\n"
            "• MERGE: Equivalent methods with different names (e.g., 'control sample calibration' ≈ 'data-driven calibration')\n"
            "• MERGE: Same method at different levels of sophistication (e.g., 'simple reweighting' and 'kinematic reweighting')\n"
            "• MERGE: Variations of the same core approach (e.g., 'fit variation with polynomial background' and 'fit variation with exponential background')\n"
            "• DO NOT MERGE: Fundamentally different methodological approaches (Monte Carlo vs data-driven vs theoretical calculation)\n"
            "• DO NOT MERGE: Different evaluation philosophies (conservative assignment vs detailed modeling vs external input)\n"
            "• FOCUS ON: The methodological approach rather than the specific physics target"
        ),
    }
}

import os
NEO4J_CONFIG = {
    "uri" : os.getenv("NEO4J_URI"),
    "username" : os.getenv("NEO4J_USERNAME"),
    "password" : os.getenv("NEO4J_PASSWORD"),
}

