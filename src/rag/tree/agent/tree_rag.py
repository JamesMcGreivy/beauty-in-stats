"""
Author: James McGreivy
Email: mcgreivy@mit.edu

Hierarchical RAG implementation using TreeRAG:
- TreeRAG: Hierarchical tree search with attention-based embedding diffusion
- Automatically loads pre-computed paper trees and embeddings from cache
- Enhanced with incremental hierarchical output formatting and token-based limits
"""

import numpy as np
import os
import time
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path
import pickle
from tqdm import tqdm

import rag.settings as settings
from rag.tree.core.paper_tree import PaperTree
from rag.tree.core.utils import load_paper_trees

def get_paper_id(paper):
    """Generate hierarchical ID for a paper based on its position in the tree"""
    if not paper.parent:
        return paper.title
    else:
        return f"{get_paper_id(paper.parent)} --> {paper.title}"


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Compute softmax with optional temperature scaling and numerical stability"""
    x = np.array(x) / temperature
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


def validate_paper_trees(papers: List[PaperTree]) -> None:
    """
    Validate that paper trees are properly summarized and have embeddings.
    """
    if not papers:
        raise Exception(
            "No papers loaded from cache.\n"
            "Please run 'python /rag/tree/run.py' to build the paper tree cache."
        )
    
    total_sections = 0
    sections_without_summaries = 0
    sections_without_embeddings = 0
    
    def check_recursive(paper: PaperTree) -> None:
        nonlocal total_sections, sections_without_summaries, sections_without_embeddings
        total_sections += 1
        
        # Check for missing abstracts (summaries)
        if paper.abstract is None or not paper.abstract.strip():
            if len(paper.sections) > 0:  # Only count non-leaf nodes
                sections_without_summaries += 1
        
        # Check for missing embeddings
        if not hasattr(paper, 'embedding') or paper.embedding is None:
            if paper.abstract and paper.abstract.strip():  # Only count sections that should have embeddings
                sections_without_embeddings += 1
        
        # Recursively check all subsections
        for section in paper.sections:
            check_recursive(section)
    
    for paper in papers:
        check_recursive(paper)
    
    print(f"Validation results: {total_sections} total sections")
    
    # Check for missing summaries
    if sections_without_summaries > 0:
        raise Exception(
            f"Found {sections_without_summaries} sections without summaries.\n"
            f"Please run 'python /rag/tree/run.py' to complete the summarization process."
        )
    
    # Check for missing embeddings
    if sections_without_embeddings > 0:
        raise Exception(
            f"Found {sections_without_embeddings} sections without embeddings.\n"
            f"Please run 'python /rag/tree/run.py' to complete the embedding process."
        )
    
    print("All paper trees are properly summarized and embedded!")


class HierarchicalOutput:
    """Tracks hierarchical output with section-level summaries and token count incrementally during search"""
    
    def __init__(self, embedding_model, max_tokens):
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        self.remaining_tokens = max_tokens
        self.output_paths = {"children" : {}}
        
    def _count_tokens(self, text: str) -> int:
        """Count tokens using the embedding model's tokenizer"""
        tokens = self.embedding_model.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    
    def _count_path_tokens(self, path: dict) -> int:
        """Count the number of tokens in a full path"""
        paper = path["paper"]
        children = path["children"]
        
        # Include both title and abstract in token count
        depth = paper.get_depth()
        indent = "\t" * depth
        line = f"{indent}{paper.title}: {paper.abstract}"
        
        return self._count_tokens(line) + sum([self._count_path_tokens(child) for child in children.values()])
    
    def add_result(self, paper) -> bool:
        """
        Add a result to the hierarchical output with full path summaries.
        
        Returns: bool succesful or unsuccesful (output buffer is full)
        """

        current_paper = paper
        children = {}

        while current_paper:
            input_path = {"paper" : current_paper, "children" : children}
            idx = current_paper.parent.sections.index(current_paper) if current_paper.parent else current_paper.title
            children = {idx : input_path}
            current_paper = current_paper.parent
    
        output_paths = self.output_paths

        while children:
            idx, input_path = list(children.items())[0]

            if idx in output_paths["children"].keys():
                output_paths = output_paths["children"][idx]
                children = input_path["children"]

            else:
                new_tokens = self._count_path_tokens(input_path)
                if new_tokens < self.remaining_tokens:
                    self.remaining_tokens -= new_tokens
                    output_paths["children"][idx] = input_path
                    return True
                else:
                    return False
                
        return True
    
    def get_output(self) -> str:
        """Get the final formatted output by traversing the tree structure"""
        output_lines = []
        self._format_recursive(self.output_paths, output_lines, depth=-1)
        return "\n".join(output_lines)

    def _format_recursive(self, node: dict, output_lines: list, depth: int):
        """Recursively format the tree into output lines"""
        # Skip the root wrapper node (depth = -1)
        if depth >= 0:
            paper = node["paper"]
            indent = "\t" * depth
            output_lines.append(f"{indent}{paper.title}: {paper.abstract}")
            output_lines.append("")  # Empty line after each entry
        
        # Process children, sorted by their index
        children = node["children"]
        if children:
            # Sort children - integers first (section indices), then strings (root titles)
            sorted_indices = sorted(children.keys(), key=lambda x: (isinstance(x, str), x))
            
            for idx in sorted_indices:
                child = children[idx]
                self._format_recursive(child, output_lines, depth + 1)
    

### Tree RAG Implementation
class TreeRAG:
    """
    Hierarchical RAG that uses tree structure to intelligently navigate to relevant content.
    Uses attention-based embedding diffusion to create semantically meaningful parent embeddings.
    
    Automatically loads paper trees from cache upon instantiation.
    """

    # Class variables for shared model (lazy loading)
    _embedding_model = None
    
    @classmethod
    def _get_embedding_model(cls):
        """
        Get the shared embedding model, loading it if necessary.
        """
        if cls._embedding_model is None:
            model = settings.EMBEDDING_CONFIG["model"]
            device = settings.EMBEDDING_CONFIG["device"]

            from sentence_transformers import SentenceTransformer
            print(f"Loading embedding model: {model} on {device}")
            cls._embedding_model = SentenceTransformer(model, device=device)
        
        return cls._embedding_model
    
    def __init__(self, l: float = 0.5, temperature: float = 1.0, cache_dir: str = settings.TREE_RAG_CACHE):
        """
        Initialize TreeRAG by loading paper trees from cache.
        
        Args:
            l: Blending parameter for embedding diffusion (0=children only, 1=parent only)
            temperature: Temperature for attention softmax
        """
        self.unique_id = str(int(10000 * time.time()) % 2**16)     
        
        self.embedding_model = self._get_embedding_model()
        
        print(f"Initializing TreeRAG from cache: {cache_dir}")
        
        # Load paper trees from cache using utils
        try:
            self.papers = load_paper_trees(cache_dir, show_progress=True)
        except Exception as e:
            raise Exception(
                f"Unexpected error loading paper trees: {e}\n"
                f"Please run 'python /rag/tree/run.py' to build the cache."
            )
        
        # Validate that papers are properly processed
        validate_paper_trees(self.papers)

        # Initialize internal data structures
        self.id_to_paper = {}
        self.level_zero = []  # Root-level papers (entry points for search)
        
        # Build the paper tree index
        for paper in tqdm(self.papers, desc="Indexing papers"):
            self.add_paper(paper)

        # Perform embedding diffusion to create meaningful parent embeddings
        for paper in tqdm(self.papers, desc="Diffusing embeddings"):
            self._embedding_diffusion_attention(paper, l, temperature)

        # Create fast lookup for embeddings during search
        self.id_to_embedding = {id: paper.diffuse_embedding for id, paper in self.id_to_paper.items()}

    def add_paper(self, paper):
        """Build the paper tree index and identify root nodes"""
        id = get_paper_id(paper)
        self.id_to_paper[id] = paper
        
        # Track root-level papers as starting points for tree search
        if not paper.parent:
            self.level_zero.append(id)
        
        # Recursively add all sections
        for section in paper.sections:
            self.add_paper(section)

    def _embedding_diffusion_attention(self, paper, l: float = 0.5, temperature: float = 1.0):
        """
        Create parent embeddings using attention-weighted averaging of children.
        Parents give more weight to semantically similar children, creating more
        coherent representations for tree navigation.
        """
        if not paper.sections:
            # Leaf nodes: use original embedding unchanged
            paper.diffuse_embedding = np.copy(paper.embedding)
            return paper.diffuse_embedding
        
        # Recursively process children first (bottom-up diffusion)
        child_embeddings = []
        for section in paper.sections:
            child_emb = self._embedding_diffusion_attention(section, l, temperature)
            child_embeddings.append(child_emb)
        
        # Compute attention weights based on cosine similarity to parent
        parent_embedding = paper.embedding / np.linalg.norm(paper.embedding)
        
        similarities = []
        for child_emb in child_embeddings:
            child_emb_norm = child_emb / np.linalg.norm(child_emb)
            similarity = np.dot(parent_embedding, child_emb_norm)
            similarities.append(similarity)
        
        # Convert similarities to attention weights using softmax
        attention_weights = softmax(similarities, temperature)
        
        # Compute attention-weighted average of children
        weighted_child_avg = np.zeros_like(child_embeddings[0])
        for weight, child_emb in zip(attention_weights, child_embeddings):
            weighted_child_avg += weight * child_emb
        
        # Blend parent's original embedding with weighted children average
        diffuse_embedding = l * paper.embedding + (1-l) * weighted_child_avg
        
        # Normalize for consistent cosine similarity computation
        paper.diffuse_embedding = diffuse_embedding / np.linalg.norm(diffuse_embedding)
        
        return paper.diffuse_embedding

    def _compute_similarities(self, query_embedding, candidate_ids):
        """Efficiently compute cosine similarities for a batch of candidates"""
        if not candidate_ids:
            return {}
        
        # Stack embeddings for vectorized computation
        candidate_embeddings = np.stack([self.id_to_embedding[id] for id in candidate_ids])
        similarities = np.dot(candidate_embeddings, query_embedding)

        return dict(zip(candidate_ids, similarities))

    def query(self, query: str, max_tokens: int = 5000, k: int = 3) -> str:
        """
        Perform hierarchical tree search to find relevant papers with incremental output building.
        
        Args:
            query: Search query string
            max_tokens: Maximum number of tokens to include in output
            k: Number of top candidates to explore at each level
            
        Returns:
            Hierarchically formatted string with summaries at each level
        
        Algorithm:
        1. Start at root papers
        2. At each level, compute similarities for all candidates
        3. Select top-k most similar papers
        4. For each top-k paper:
           - If it has children: add children to next exploration frontier
           - If it's a leaf: check if we can add to output without exceeding token limit
             - If yes: add to incremental output
             - If no: terminate search and return current output
        5. Continue until token limit reached or tree exhausted
        """
        try:
            query_embedding = self.embedding_model.encode(query)
            
            frontier = list(self.level_zero)  # Papers to explore next
            candidates = {}  # Papers we've computed similarities for
            output_builder = output_builder = HierarchicalOutput(self.embedding_model, max_tokens)
            
            done = False
            while not done:
                # Compute similarities for current frontier and add to candidates
                if frontier:
                    frontier_distances = self._compute_similarities(query_embedding, frontier)
                    candidates.update(frontier_distances)
                    frontier = []

                # Select the top-k most promising candidates to explore
                if not candidates:
                    break
                    
                top_k_ids = sorted(candidates, key=candidates.get, reverse=True)[:k]

                # Process each of the top-k candidates
                for id in top_k_ids:
                    similarity_score = candidates.pop(id)  # Remove from candidate pool and get score
                    best_paper = self.id_to_paper[id]

                    if best_paper.sections:
                        # Internal node: add children to frontier for next iteration
                        for section in best_paper.sections:
                            frontier.append(get_paper_id(section))
                    else:
                        # Leaf node: check if we can add it to the output builder
                        success = output_builder.add_result(best_paper)
                        if not success:
                            # Token limit would be exceeded, terminate search
                            done = True
                            break

                # Safety check: stop if no more candidates and no frontier
                if not candidates and not frontier:
                    done = True
            
            return output_builder.get_output()
        
        except Exception as e:
            error_msg = f"Error while processing query: {e}"
            return error_msg