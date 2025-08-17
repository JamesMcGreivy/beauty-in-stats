"""
Compute Text Embeddings for Paper Trees

This script loads all paper trees from the cache, computes embeddings for sections
that have abstracts but are missing embeddings, and saves the updated trees back
to the cache. Supports both initial embedding computation and incremental updates.

Author: James McGreivy
Email: mcgreivy@mit.edu
"""

import time
from typing import List, Optional
from tqdm import tqdm

import rag.settings as settings
from rag.tree.core.paper_tree import PaperTree
from rag.tree.core.utils import load_paper_trees, save_paper_trees

from sentence_transformers import SentenceTransformer


class TreeEmbeddingProcessor:
    """Handles embedding computation and management for paper trees"""
    
    def __init__(self, model_name: str = settings.EMBEDDING_CONFIG["model"], device: str = "cuda"):
        """
        Initialize the embedding processor with a specified model.
        
        Args:
            model_name: Name of the sentence transformer model to use
            device: Device to run the model on ('cuda', 'cpu', etc.)
        """
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = self._get_embedding_model()
        print(f"Model loaded on device: {device}")
    
    def needs_embedding(self, paper: PaperTree) -> bool:
        """
        Check if a paper section needs an embedding computed.
        
        A section needs an embedding if:
        1. It has an abstract (content to embed)
        2. It doesn't already have an embedding, OR overwrite is enabled
        """
        # Must have abstract content to embed
        if paper.abstract is None or not paper.abstract.strip():
            return False
        
        # Check overwrite setting
        if settings.EMBEDDING_CONFIG.get("overwrite_existing", False):
            return True
        
        # Only compute if embedding doesn't exist
        return not hasattr(paper, 'embedding') or paper.embedding is None
    
    def collect_papers_needing_embeddings(self, papers: List[PaperTree]) -> List[PaperTree]:
        """
        Recursively collect all paper sections that need embeddings computed.
        
        Returns a flat list of all sections across all paper trees that need
        embedding computation.
        """
        needs_embedding = []
        
        def collect_recursive(paper: PaperTree) -> None:
            """Recursively traverse the tree and collect papers needing embeddings"""
            if self.needs_embedding(paper):
                needs_embedding.append(paper)
            
            # Process all subsections
            for section in paper.sections:
                collect_recursive(section)
        
        # Process all paper trees
        for paper in papers:
            collect_recursive(paper)
        
        return needs_embedding
    
    def compute_embeddings_batch(self, papers: List[PaperTree], batch_size: int = 1000) -> int:
        """
        Compute embeddings for a list of papers in batches for memory efficiency.
        
        Args:
            papers: List of paper sections to process
            batch_size: Number of papers to process in each batch
            
        Returns:
            Number of embeddings successfully computed
        """
        if not papers:
            return 0
        
        print(f"Computing embeddings for {len(papers)} sections...")
        successful_count = 0
        
        # Process in batches to manage memory usage
        for i in tqdm(range(0, len(papers), batch_size), desc="Embedding batches"):
            batch_papers = papers[i:i + batch_size]
            
            try:
                # Extract abstracts for batch processing
                batch_abstracts = [paper.abstract for paper in batch_papers]
                
                # Compute embeddings for the entire batch
                batch_embeddings = self.embedding_model.encode(
                    batch_abstracts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                # Assign embeddings back to papers
                for paper, embedding in zip(batch_papers, batch_embeddings):
                    paper.embedding = embedding
                    successful_count += 1
                    
            except Exception as e:
                print(f"Failed to process batch {i//batch_size + 1}: {e}")
                # Try to process papers individually if batch fails
                for paper in batch_papers:
                    try:
                        paper.embedding = self.embedding_model.encode(paper.abstract, convert_to_numpy=True)
                        successful_count += 1
                    except Exception as e2:
                        print(f"Failed to embed individual paper '{getattr(paper, 'title', 'Unknown')}': {e2}")
        
        return successful_count


def get_embedding_statistics(papers: List[PaperTree]) -> dict:
    """
    Compute statistics about embeddings in the paper collection.
    
    Returns a dictionary with counts of sections with/without embeddings.
    """
    stats = {
        'total_sections': 0,
        'sections_with_abstracts': 0,
        'sections_with_embeddings': 0,
        'sections_needing_embeddings': 0
    }
    
    def count_recursive(paper: PaperTree) -> None:
        stats['total_sections'] += 1
        
        if paper.abstract and paper.abstract.strip():
            stats['sections_with_abstracts'] += 1
            
            if hasattr(paper, 'embedding') and paper.embedding is not None:
                stats['sections_with_embeddings'] += 1
            else:
                stats['sections_needing_embeddings'] += 1
        
        for section in paper.sections:
            count_recursive(section)
    
    for paper in papers:
        count_recursive(paper)
    
    return stats


def main():
    """Main function to orchestrate the embedding computation process"""
    
    # Configuration
    cache_dir = str(settings.TREE_RAG_CACHE)
    batch_size = settings.EMBEDDING_CONFIG.get("batch_size", 1000)
    model_name = settings.EMBEDDING_CONFIG.get("model", "BAAI/bge-small-en-v1.5")
    device = settings.EMBEDDING_CONFIG.get("device", "cuda")
    
    print("=== Tree Text Embedding Computation ===")
    print(f"Cache directory: {cache_dir}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    
    # Load all paper trees using utils
    papers = load_paper_trees(cache_dir, show_progress=True)
    if not papers:
        print("No papers loaded. Run the paper tree builder first.")
        return
    
    # Show initial statistics
    initial_stats = get_embedding_statistics(papers)
    print(f"\nInitial Statistics:")
    print(f"  Total sections: {initial_stats['total_sections']}")
    print(f"  Sections with abstracts: {initial_stats['sections_with_abstracts']}")
    print(f"  Sections with embeddings: {initial_stats['sections_with_embeddings']}")
    print(f"  Sections needing embeddings: {initial_stats['sections_needing_embeddings']}")
    
    # Check if any work needs to be done
    if initial_stats['sections_needing_embeddings'] == 0:
        overwrite_enabled = settings.EMBEDDING_CONFIG.get("overwrite_existing", False)
        if not overwrite_enabled:
            print("\nAll sections already have embeddings!")
            print("Set EMBEDDING_CONFIG['overwrite_existing'] = True to recompute all embeddings.")
            return
        else:
            print("\nOverwrite enabled - recomputing all embeddings...")
    
    # Initialize embedding processor
    processor = TreeEmbeddingProcessor(model_name=model_name, device=device)
    
    # Find papers that need embeddings
    papers_to_embed = processor.collect_papers_needing_embeddings(papers)
    print(f"\nFound {len(papers_to_embed)} sections to process")
    
    if not papers_to_embed:
        print("No sections need embedding computation. Exiting.")
        return
    
    # Compute embeddings
    start_time = time.time()
    successful_embeddings = processor.compute_embeddings_batch(papers_to_embed, batch_size)
    elapsed_time = time.time() - start_time
    
    print(f"\nEmbedding computation complete:")
    print(f"  Processed: {successful_embeddings}/{len(papers_to_embed)} sections")
    print(f"  Time elapsed: {elapsed_time:.2f} seconds")
    if elapsed_time > 0:
        print(f"  Rate: {successful_embeddings/elapsed_time:.2f} embeddings/second")
    
    # Save updated paper trees using utils
    if successful_embeddings > 0:
        print("\nSaving updated paper trees...")
        saved_count = save_paper_trees(papers, cache_dir, show_progress=True)
        
        # Show final statistics
        final_stats = get_embedding_statistics(papers)
        print(f"\nFinal Statistics:")
        print(f"  Sections with embeddings: {final_stats['sections_with_embeddings']}")
        print(f"  Sections still needing embeddings: {final_stats['sections_needing_embeddings']}")
        
        print(f"\n Successfully processed {successful_embeddings} embeddings and saved {saved_count} papers")
    else:
        print("\n No embeddings were computed successfully")


if __name__ == "__main__":
    main()