"""
Paper Tree Utilities

Shared utilities for loading, saving, and processing paper trees across
the TreeRAG system. Provides consistent file handling and error management.

Author: James McGreivy
Email: mcgreivy@mit.edu
"""

import os
import pickle
from pathlib import Path
from typing import List, Optional, Callable
from tqdm import tqdm

import rag.settings as settings
from rag.tree.core.paper_tree import PaperTree


def generate_filename(paper_title: str) -> str:
    """
    Generate a consistent .pkl filename from a paper title.
    
    Args:
        paper_title: The title/name of the paper
        
    Returns:
        Standardized .pkl filename
    """
    base_name = Path(paper_title).stem  # Remove any existing extension
    return f"{base_name}.pkl"


def load_paper_trees(
    cache_dir: str = settings.TREE_RAG_CACHE, 
    filter_func: Optional[Callable[[str], bool]] = None,
    show_progress: bool = True
) -> List[PaperTree]:
    """
    Load paper trees from the cache directory with optional filtering.
    
    Args:
        cache_dir: Directory containing .pkl files
        filter_func: Optional function to filter which files to load based on filename
        show_progress: Whether to show progress bar
        
    Returns:
        List of successfully loaded PaperTree objects
        
    Raises:
        FileNotFoundError: If cache directory doesn't exist
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache directory does not exist: {cache_dir}")
    
    print(f"Loading paper trees from: {cache_dir}")
    
    # Find all pickle files
    pkl_files = list(cache_path.glob("*.pkl"))
    if not pkl_files:
        print("No .pkl files found in cache directory")
        return []
    
    # Apply filter if provided
    if filter_func:
        pkl_files = [f for f in pkl_files if filter_func(f.name)]
        if not pkl_files:
            print("No files match the filter criteria")
            return []
    
    papers = []
    failed_files = []
    
    # Set up progress bar
    iterator = tqdm(pkl_files, desc="Loading papers") if show_progress else pkl_files
    
    for pkl_file in iterator:
        try:
            with open(pkl_file, 'rb') as f:
                paper = pickle.load(f)
                papers.append(paper)
        except Exception as e:
            print(f"Failed to load {pkl_file.name}: {e}")
            failed_files.append(pkl_file.name)
    
    print(f"Successfully loaded {len(papers)} papers")
    if failed_files:
        print(f"Failed to load {len(failed_files)} papers: {failed_files[:5]}{'...' if len(failed_files) > 5 else ''}")
    
    return papers


def save_paper_trees(
    papers: List[PaperTree], 
    cache_dir: str,
    show_progress: bool = True,
    create_backup: bool = False
) -> int:
    """
    Save paper trees to the cache directory with consistent filename handling.
    
    Args:
        papers: List of paper trees to save
        cache_dir: Directory to save the trees to
        show_progress: Whether to show progress bar
        create_backup: Whether to create .bak files before overwriting
        
    Returns:
        Number of papers successfully saved
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    if show_progress:
        print(f"Saving {len(papers)} paper trees to: {cache_dir}")
    
    successful_count = 0
    failed_papers = []
    
    # Set up progress bar
    iterator = tqdm(papers, desc="Saving papers") if show_progress else papers
    
    for paper in iterator:
        try:
            # Generate consistent filename
            title = getattr(paper, 'title', 'unknown_paper')
            filename = generate_filename(title)
            filepath = cache_path / filename
            
            # Create backup if requested and file exists
            if create_backup and filepath.exists():
                backup_path = filepath.with_suffix('.pkl.bak')
                filepath.rename(backup_path)
            
            with open(filepath, 'wb') as f:
                pickle.dump(paper, f)
            successful_count += 1
            
        except Exception as e:
            error_info = f"{getattr(paper, 'title', 'Unknown')}: {e}"
            print(f"Failed to save paper {error_info}")
            failed_papers.append(error_info)
    
    if show_progress:
        print(f"Successfully saved {successful_count}/{len(papers)} papers")
        if failed_papers:
            print(f"Failed papers: {failed_papers[:3]}{'...' if len(failed_papers) > 3 else ''}")
    
    return successful_count


def check_paper_exists(paper_title: str, cache_dir: str) -> bool:
    """
    Check if a paper already exists in the cache.
    
    Args:
        paper_title: Title/name of the paper to check
        cache_dir: Cache directory to check in
        
    Returns:
        True if the paper exists, False otherwise
    """
    cache_path = Path(cache_dir)
    filename = generate_filename(paper_title)
    filepath = cache_path / filename
    return filepath.exists()