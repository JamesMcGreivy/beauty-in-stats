"""
Author: James McGreivy
Email: mcgreivy@mit.edu
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
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
import os

import SystematicsGraph


def build_papers_cache(tex_dir, abstract_dir, cache_dir, max_papers=-1):

    os.makedirs(cache_dir, exist_ok=True)
    # Remove everything currently in the cache directory
    files = glob.glob(os.path.join(cache_dir, "*"))
    for f in files:
        os.remove(f)

    print(f"Loading papers from {tex_dir}")

    tex_files = [f for f in os.listdir(tex_dir) if f.endswith(".tex")]
    
    if max_papers != -1:
        tex_files = tex_files[:max_papers]
    
    for file in tqdm(tex_files, desc="Loading papers"):

        abstract_path = os.path.join(abstract_dir, file)
        try:
            with open(abstract_path, 'r', encoding='utf-8') as f:
                abstract = f.read()
        except Exception as e:
            tqdm.write(f"Warning: Abstract not found for {file} \n {e}")
            continue

        tex_path = os.path.join(tex_dir, file)
        with open(tex_path, 'r', encoding='utf-8') as f:
            text = f.read()

        arxiv_id = file[:file.find(".tex")]

        paper = SystematicsGraph.LHCbPaper(abstract, text, arxiv_id)
        
        with open(os.path.join(cache_dir, paper.arxiv_id + ".pkl"), "wb") as f:
            pickle.dump(paper, f)
            

async def process_papers_async(cache_dir, threads=6, timeout=360):
    
    files = list(filter(lambda x : ".pkl" in x, os.listdir(cache_dir)))

    if len(files) == 0:
        print("No files found in the paper cache. Have you run with --build_cache yet?")

    # Create progress bar
    pbar = tqdm(total=len(files), desc="Processing papers")
    semaphore = asyncio.Semaphore(threads)
    
    async def process_with_limit(file_path):

        with open(file_path, "rb") as f:
            paper = pickle.load(f)

        """Process a single paper with semaphore limiting."""
        if (hasattr(paper, "relationships") and hasattr(paper, "entities")) and (len(paper.relationships) > 0 or len(paper.entities) > 0):
            tqdm.write(f"Already processed {paper.arxiv_id}, skipping...")
            pbar.update(1)
            return
        
        async with semaphore:
            try:
                await asyncio.wait_for(
                    paper.process_paper(),
                    timeout=timeout
                )
                tqdm.write(f"Successfully processed: {paper.arxiv_id} \nSaving processed version to cache.")

                with open(file_path, "wb") as f:
                    pickle.dump(paper, f)

            except Exception as e:
                tqdm.write(f"Exception {e} while processing {paper.arxiv_id}")
            finally:
                pbar.update(1)
    
    tasks = [process_with_limit(os.path.join(cache_dir, file)) for file in files]        
    await asyncio.gather(*tasks)
    pbar.close()
    
    print("Finished processing all papers")

def process_papers(cache_dir, threads=6, timeout=360):
    asyncio.run(process_papers_async(cache_dir, threads, timeout))

def load_papers(cache_dir):
    papers = []

    files = list(filter(lambda x : ".pkl" in x, os.listdir(cache_dir)))

    for file in files:
        file_path = os.path.join(cache_dir, file)
        with open(file_path, "rb") as f:
            paper = pickle.load(f)
            papers.append(paper)

    return papers


def build_graph(papers):
    graph = SystematicsGraph.SystematicsGraph()
    
    for paper in tqdm(papers, desc="Loading papers into graph"):
        graph.load_paper(paper)
    
    return graph


def merge_entities(graph, stop_ratio):
    print("Merging similar entities...")
    
    # Merge uncertainty sources
    print("Merging uncertainty sources...")
    graph.merge_entity_type("uncertainty_source", ["type"], stop_ratio=stop_ratio, verbose=True)
    
    # Merge methods
    print("Merging methods...")
    graph.merge_entity_type("method", [], stop_ratio=stop_ratio, verbose=True)


def push_to_neo4j(graph, uri, username, password):
    print(f"Pushing graph to Neo4j at {uri}")
    graph.push_to_neo4j(uri, username, password)
    print("Successfully pushed graph to Neo4j")


def main():
    parser = argparse.ArgumentParser(description="Process and cache paper data.")
    parser.add_argument('--build_cache', action='store_true', help='Build or rebuild the paper cache. This will delete ALL processed')
    args = parser.parse_args()

    # Config
    tex_dir = "../scraper/data/cleaned_tex"
    abstract_dir = "../scraper/data/abstracts"
    cache_dir = "./systematics_graph_cache/papers"

    if args.build_cache:
        build_papers_cache(tex_dir, abstract_dir, cache_dir)

    process_papers(cache_dir)

    # Uncomment these if you want to use Neo4j or graph processing later
    papers = load_papers(cache_dir)
    graph = build_graph(papers)
    merge_entities(graph, 0.2)

    # Need a neo4j account
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    push_to_neo4j(graph, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

if __name__ == "__main__":
    main()  