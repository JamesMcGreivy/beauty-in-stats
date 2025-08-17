"""
Graph Extraction Pipeline: Automated knowledge graph construction from academic papers.

This module provides functionality to extract structured knowledge graphs from academic
papers using Large Language Models (LLMs). It processes paper abstracts and full text
to identify entities, relationships, and metadata, then caches the results for later
use in building comprehensive knowledge graphs.

The pipeline works in several stages:
1. Load paper trees from cached sources
2. Extract metadata from paper abstracts
3. Extract entities and relationships from paper content
4. Structure the data into a standardized graph format
5. Cache results for efficient reuse

Author: James McGreivy
Email: mcgreivy@mit.edu
"""

import numpy as np
import os
import regex as re 
import subprocess
import importlib
from collections import deque
import json
import asyncio
from tqdm import tqdm
import pickle
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import rag.settings as settings
from rag.tree import load_paper_trees
from rag.tree.core.paper_tree import PaperTree
from rag.graph.builder.prompts import get_abstract_extraction_prompt, get_graph_extraction_prompt

from llama_index.llms.openai import OpenAI

# Initialize the language model with configuration from settings
llm = OpenAI(temperature=0, model=settings.GRAPH_CONFIG["model"], timeout=240, max_retries=3,)


def get_relevant_text(paper: PaperTree) -> str:
    """
    Extract and combine relevant text sections from a paper for graph extraction.
    
    Filters out sections based on ignore keywords (like acknowledgments, references)
    and combines the abstract with relevant section content. This preprocessing
    step ensures the LLM focuses on scientifically relevant content.
    """
    section_titles = [section.title for section in paper.sections]
    section_texts = [section.text for section in paper.sections]

    # Start with the abstract as the foundation
    relevant_text = paper.abstract

    # Define sections to ignore based on configuration
    ignore_keywords = settings.GRAPH_CONFIG["section_ignore_keywords"] + settings.LATEX_ELEMENT_TYPES
    ignore_flags = [
        any([flag in title for flag in ignore_keywords]) 
        for title in section_titles
    ]
    
    # Append relevant sections to the text
    for section_title, section_text, ignore_flag in zip(section_titles, section_texts, ignore_flags):
        if not ignore_flag:
            relevant_text += f"\n{section_title}:\n{section_text}"
    
    return relevant_text


async def extract_metadata_from_abstract(abstract: str) -> Dict[str, Any]:
    """
    Extract structured metadata from a paper's abstract using an LLM.
    
    Uses a specialized prompt to identify key scientific concepts like observables,
    decay processes, and other domain-specific metadata. Falls back to a default
    structure if extraction fails.
    """
    prompt, default = get_abstract_extraction_prompt(abstract)

    response = None

    response = (await llm.acomplete(prompt)).text
    # Parse the LLM response as Python code (expected to return a dictionary)
    parsed_response = eval(response.replace("```python", "").replace("```", ""))
    return parsed_response


async def extract_graph_from_text(body_text: str, observables: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract entities and their properties from paper text using an LLM.
    
    Takes the full paper text and a list of observables (from metadata extraction)
    to generate a structured representation of entities mentioned in the paper.
    The LLM identifies different types of scientific entities and their attributes.
    """
    prompt, default = get_graph_extraction_prompt(body_text, observables)

    response = None

    response = (await llm.acomplete(prompt)).text
    # Parse the LLM response as Python code (expected to return a dictionary)
    parsed_response = eval(response.replace("```python", "").replace("```", ""))
    return parsed_response


async def extract_graph_from_paper(paper: PaperTree) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Main extraction pipeline that processes a complete paper into a structured graph.
    
    This function orchestrates the entire extraction process:
    1. Extracts metadata from the abstract
    2. Processes the full paper text for entities
    3. Restructures relationship data from entity attributes
    4. Adds paper-level entities and relationships
    5. Returns a complete graph structure ready for knowledge graph construction
    """
    # Extract high-level metadata from the abstract
    metadata = await extract_metadata_from_abstract(paper.abstract)

    # Separate out special entity types that need different handling
    decays = metadata.pop("decay", [])
    observables = metadata.pop("observable", [])
    metadata.pop("explanation", None)  # Remove explanation field if present

    # Get the relevant text content for detailed extraction
    relevant_text = get_relevant_text(paper)
    
    # Extract detailed entities from the paper content
    entities = await extract_graph_from_text(relevant_text, observables)
    
    # Add metadata entities back to the entity collection
    entities["decay"] = decays
    entities["observable"] = observables

    # Process relationship data embedded in entity attributes
    relationships: Dict[str, List[Dict[str, Any]]] = {}
    
    for entity_type in entities:
        for entity in entities[entity_type]:
            # Find all relationship attributes (prefixed with "relationship_")
            relationship_keys = [key for key in entity.keys() if key.startswith("relationship_")]
            
            for key in relationship_keys:
                relationship_type = key.removeprefix("relationship_")
                
                # Initialize relationship type if not exists
                if relationship_type not in relationships:
                    relationships[relationship_type] = []
                
                # Extract and restructure relationship data
                entity_relationships = entity.pop(key)
                for relationship in entity_relationships:
                    relationship["source"] = entity["name"]
                    relationships[relationship_type].append(relationship)

    # Add the paper itself as an entity with metadata
    entities["paper"] = [{
        "description": paper.abstract, 
        "name": paper.title, 
        **metadata
    }]

    # Create relationships between paper and its observables
    relationships["determines"] = []
    for observable in entities["observable"]:
        relationships["determines"].append({
            "source": paper.title, 
            "target": observable["name"]
        })

    # Return the complete graph structure
    paper_graph = {
        "entities": entities, 
        "relationships": relationships
    }
    return paper_graph


def get_cache_path(paper: PaperTree) -> Path:
    """
    Generate the cache file path for a processed paper graph.
    
    Creates a standardized file path based on the paper's identifier,
    ensuring the cache directory structure exists.
    """
    cache_dir = Path(settings.GRAPH_RAG_CACHE) / "papers"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract arxiv ID from the paper title (assuming it's a file path)
    arxiv_id = Path(paper.title).stem
    return cache_dir / f"{arxiv_id}.pkl"  # Fixed: was using 'dir' instead of 'cache_dir'

async def process_single_paper(paper: PaperTree, semaphore: asyncio.Semaphore):

    overwrite = settings.GRAPH_CONFIG.get("overwrite_existing", False)
    cache_path = get_cache_path(paper)

    # Skip processing if cached result exists and overwrite is disabled
    if cache_path.exists() and not overwrite:
        return 0, 1

    # Extract graph structure from the paper
    async with semaphore:
        try:
            paper_graph = await extract_graph_from_paper(paper)
        except Exception as e:
            print(f"Exception occured while processing paper into graph:\n{e}\nSkipping...")
            return 0, 1

    # Cache the result for future use
    with open(cache_path, "wb") as f:
        pickle.dump(paper_graph, f)

    return 1, 0


async def async_main():
    """
    Main asynchronous processing function that handles the complete pipeline.
    
    Loads all available papers, processes them through the graph extraction
    pipeline, and caches the results. Provides progress tracking and handles
    both fresh processing and cached result reuse based on configuration.
    """
    # Load all available paper trees
    papers = load_paper_trees()
    if not papers:
        print("No papers returned by load_paper_trees(). Make sure to run rag.tree.build_cache() first.")
        return
    
    max_threads = settings.GRAPH_CONFIG.get("max_threads")
    semaphore = asyncio.Semaphore(max_threads)
    
    # Create tasks for all papers
    tasks = [process_single_paper(paper, semaphore) for paper in papers]
    
    from tqdm.asyncio import tqdm
    results = await tqdm.gather(*tasks, desc="Building graphs")
    
    # Calculate and report final statistics
    wrote = sum(r[0] for r in results)
    skipped = sum(r[1] for r in results)
    
    print(f"Done. wrote={wrote}, skipped={skipped}")


def main():
    """
    Entry point for the graph extraction pipeline.
    
    Runs the asynchronous main function to process all papers and build
    their corresponding knowledge graph representations.
    """
    asyncio.run(async_main())


if __name__ == "__main__":
    main()