"""
TreeRAG Hierarchical Paper Summarization System

This module creates hierarchical summaries for scientific papers in a TreeRAG system.
It processes paper trees depth-by-depth, starting from the deepest level and working
upward, generating concise, keyword-rich summaries that enable precise semantic 
similarity matching during retrieval.

Author: James McGreivy
Email: mcgreivy@mit.edu
"""

import regex as re
import asyncio
import time
from typing import List, Optional, Dict, Set
import nest_asyncio
import asyncio

import rag.settings as settings
from rag.tree.core.paper_tree import PaperTree
from rag.tree.core.utils import load_paper_trees, save_paper_trees

from llama_index.llms.openai import OpenAI

# Initialize LLM
llm = OpenAI(temperature=0, model=settings.SUMMARY_CONFIG["model"])

# Rate limiting configuration to prevent API overload
SEMAPHORE = asyncio.Semaphore(settings.SUMMARY_CONFIG["max_threads"])  # Reduced for safety


def get_paper_abstract(paper: PaperTree) -> str:
    """
    Get the root paper's abstract by traversing up the tree.
    This provides domain context for section summarization.
    """
    current = paper
    while current.parent is not None:
        current = current.parent
    return current.abstract


def get_summary_prompt(excerpt: str, paper_abstract: str, section_title: Optional[str] = None) -> str:
    """
    Generate a domain-agnostic summarization prompt that creates concise, keyword-rich summaries.
    
    Creates prompts optimized for semantic similarity matching in tree traversal,
    focusing on key terms and concepts rather than verbose descriptions.
    """

    prompt = f"""
GOAL:
Create a concise summary (less than 200 tokens) that captures the essential content of this section. Focus on technical terms, methodologies, measurements, and key results that researchers would search for. You will be given the abstract of the full paper for your own context, as well as the specific content to be summarized.

AVOID:
- Generic introductory phrases ("This section discusses...")
- Filler words
- Basic background information
- Verbose phrasings
- Numerical values

PAPER CONTEXT:
{paper_abstract}

### Content to Summarize:
{excerpt}

### Keyword-Rich Summary (50-100 tokens):

    """

    return prompt


def calculate_depth(paper: PaperTree, current_depth: int = 0) -> int:
    """
    Calculate the depth of a paper tree node.
    Root nodes are at depth 0, their children at depth 1, etc.
    """
    if not paper.sections:  # Leaf node
        return current_depth
    
    max_child_depth = current_depth
    for section in paper.sections:
        child_depth = calculate_depth(section, current_depth + 1)
        max_child_depth = max(max_child_depth, child_depth)
    
    return max_child_depth


def find_max_depth(papers: List[PaperTree]) -> int:
    """
    Find the maximum depth across all paper trees.
    """
    max_depth = 0
    for paper in papers:
        paper_depth = calculate_depth(paper)
        max_depth = max(max_depth, paper_depth)
    return max_depth


def collect_nodes_at_depth(papers: List[PaperTree], target_depth: int) -> List[PaperTree]:
    """
    Collect all nodes at a specific depth across all paper trees.
    """
    nodes_at_depth = []
    
    def traverse_and_collect(paper: PaperTree, current_depth: int = 0) -> None:
        if current_depth == target_depth:
            nodes_at_depth.append(paper)
            return  # Don't traverse deeper once we reach target depth
        
        # Continue traversing to children if we haven't reached target depth
        for section in paper.sections:
            traverse_and_collect(section, current_depth + 1)
    
    for paper in papers:
        traverse_and_collect(paper)
    
    return nodes_at_depth


def is_leaf_node(paper: PaperTree) -> bool:
    """
    Check if a paper tree node is a leaf (has no sections).
    """
    return len(paper.sections) == 0


def initialize_leaf_summaries(nodes: List[PaperTree]) -> int:
    """
    Initialize summaries for leaf nodes using their text content.
    Returns the number of nodes processed.
    """
    processed_count = 0
    for node in nodes:
        if is_leaf_node(node):
            if not node.text:
                print(f"Warning: Leaf node '{getattr(node, 'title', 'Unknown')}' has no text content")
                node.abstract = ""
            else:
                node.abstract = node.text
            processed_count += 1
    return processed_count


async def summarize_with_retry(paper: PaperTree, max_retries: int = 3) -> str:
    """
    Generate concise, keyword-rich summary for a paper section with retry logic.
    
    Uses rate limiting and retry mechanisms to handle API failures gracefully.
    Aggregates content from all subsections and provides paper context for better summarization.
    """
    async with SEMAPHORE:
        request_delay = settings.SUMMARY_CONFIG["request_delay"]
        await asyncio.sleep(request_delay)
        
        # Get paper abstract for domain context
        paper_abstract = get_paper_abstract(paper)
        
        # Aggregate content from all child sections
        text = ""
        for section in paper.sections:
            if section.abstract:
                # Handle special elements like figures and tables
                elements = ["figure", "table", "sidewaystable"]
                if any([element in section.title.lower() for element in elements]):
                    text += f"{section.title} caption: {section.abstract}\n"
                else:
                    text += f"{section.abstract}\n"
            else:
                print(f"Warning: Child section '{getattr(section, 'title', 'Unknown')}' has no abstract")
        
        if not text.strip():
            print(f"Warning: No content to summarize for '{getattr(paper, 'title', 'Unknown')}'")
            return ""
        
        # Extract section title for contextual prompting
        section_title = getattr(paper, 'title', None)
        
        # Retry loop with exponential backoff
        for attempt in range(max_retries):
            try:
                result = await llm.acomplete(get_summary_prompt(text, paper_abstract, section_title=section_title))
                return result.text.strip()
            except Exception as e:
                wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
                print(f"Attempt {attempt + 1} failed for paper {section_title}: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Failed to summarize after {max_retries} attempts")
                    raise e


async def process_non_leaf_batch(nodes: List[PaperTree]) -> int:
    """
    Process a batch of non-leaf nodes concurrently with proper error handling.
    
    Returns the number of successfully processed nodes. Failed summarizations
    are logged but don't stop processing of other nodes in the batch.
    """
    if not nodes:
        return 0
    
    print(f"Processing batch of {len(nodes)} non-leaf nodes...")
    
    tasks = [summarize_with_retry(node) for node in nodes]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_count = 0
    for node, result in zip(nodes, results):
        if isinstance(result, Exception):
            print(f"Failed to summarize node '{getattr(node, 'title', 'Unknown')}': {result}")
            continue
        node.abstract = result
        successful_count += 1
    
    print(f"Successfully processed {successful_count}/{len(nodes)} non-leaf nodes")
    return successful_count


def nodes_ready_for_summarization(nodes: List[PaperTree]) -> List[PaperTree]:
    """
    Filter nodes that are ready for summarization.
    A node is ready if:
    - It's a leaf node (will use text as summary), OR
    - It's a non-leaf node and all its children have summaries, AND
    - It doesn't already have a summary (unless overwrite is enabled), AND
    - It's not a root node (depth 0)
    """
    ready_nodes = []
    overwrite_enabled = settings.SUMMARY_CONFIG.get("overwrite_existing", False)
    
    for node in nodes:
        # Skip root nodes (depth 0) - they keep their original abstracts
        if node.parent is None:
            continue
        
        # If overwrite is disabled and node already has abstract, skip
        if not overwrite_enabled and node.abstract is not None:
            continue
        
        # Leaf nodes are always ready (they use their text)
        if is_leaf_node(node):
            ready_nodes.append(node)
            continue
        
        # Non-leaf nodes are ready if all children have abstracts
        all_children_ready = True
        for child in node.sections:
            if child.abstract is None:
                all_children_ready = False
                break
        
        if all_children_ready:
            ready_nodes.append(node)
    
    return ready_nodes


async def async_main():
    """
    Main async processing function that orchestrates the depth-first hierarchical summarization.
    
    Processes papers depth-by-depth starting from the deepest level and working upward.
    Saves progress after each depth level to enable crash recovery.
    """
    cache_dir = settings.TREE_RAG_CACHE

    # Load all paper trees from disk using utils
    papers = load_paper_trees(cache_dir, show_progress=True)
    
    if not papers:
        print("No papers found in cache. Run the paper tree builder first.")
        return
    
    # Find the maximum depth across all paper trees
    max_depth = find_max_depth(papers)
    print(f"Maximum tree depth found: {max_depth}")
    
    if max_depth == 0:
        print("All trees have depth 0 (only root nodes). No summarization needed.")
        return
    
    try:
        # Process each depth level from deepest to shallowest (excluding depth 0)
        for current_depth in range(max_depth, 0, -1):  # max_depth down to 1
            print(f"\n=== Processing Depth {current_depth} ===")
            
            # Collect all nodes at current depth
            nodes_at_depth = collect_nodes_at_depth(papers, current_depth)
            print(f"Found {len(nodes_at_depth)} nodes at depth {current_depth}")
            
            if not nodes_at_depth:
                continue
            
            # Filter nodes that are ready for summarization
            ready_nodes = nodes_ready_for_summarization(nodes_at_depth)
            print(f"{len(ready_nodes)} nodes ready for summarization")
            
            if not ready_nodes:
                continue
            
            # Separate leaf and non-leaf nodes
            leaf_nodes = [node for node in ready_nodes if is_leaf_node(node)]
            non_leaf_nodes = [node for node in ready_nodes if not is_leaf_node(node)]
            
            # Process leaf nodes (initialize with their text)
            if leaf_nodes:
                leaf_count = initialize_leaf_summaries(leaf_nodes)
                print(f"Initialized {leaf_count} leaf node summaries")
            
            # Process non-leaf nodes in batches using LLM
            if non_leaf_nodes:
                batch_size = settings.SUMMARY_CONFIG["batch_size"]
                total_successful = 0
                
                for i in range(0, len(non_leaf_nodes), batch_size):
                    batch = non_leaf_nodes[i:i + batch_size]
                    batch_successful = await process_non_leaf_batch(batch)
                    total_successful += batch_successful
                    
                    # Save progress after each batch for crash recovery
                    saved_count = save_paper_trees(papers, cache_dir, show_progress=False)
                    print(f"Progress saved after batch {i//batch_size + 1} ({saved_count} papers updated)")
                
                print(f"Depth {current_depth} LLM processing complete: {total_successful}/{len(non_leaf_nodes)} non-leaf nodes successfully summarized")
            
            # Save progress after completing each depth level
            final_saved = save_paper_trees(papers, cache_dir, show_progress=False)
            print(f"Depth {current_depth} completed. Progress saved ({final_saved} papers)")
        
        # Final save to ensure all data is persisted
        final_saved = save_paper_trees(papers, cache_dir, show_progress=True)
        print(f"\nAll papers saved to {cache_dir} ({final_saved} papers)")
        print("Depth-first hierarchical summarization completed!")
    
    finally:
        # Properly close the LLM client to prevent event loop issues
        try:
            if hasattr(llm, 'client') and hasattr(llm.client, 'close'):
                await llm.client.close()
            elif hasattr(llm, '_client') and hasattr(llm._client, 'aclose'):
                await llm._client.aclose()
        except Exception as e:
            print(f"Warning: Could not properly close LLM client: {e}")
            pass


def main():
    """Entry point for the summarization system."""
    print("=== TreeRAG Depth-First Hierarchical Summarization ===")
    print(f"Cache directory: {settings.TREE_RAG_CACHE}")
    print(f"Batch size: {settings.SUMMARY_CONFIG['batch_size']}")
    print(f"Request delay: {settings.SUMMARY_CONFIG['request_delay']}s")
    
    overwrite_enabled = settings.SUMMARY_CONFIG.get("overwrite_existing", False)
    print(f"Overwrite existing summaries: {overwrite_enabled}")

    # Apply nest_asyncio to handle Jupyter compatibility
    nest_asyncio.apply()
    
    # Create a new event loop to ensure proper blocking
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # This will definitely block until completion
        loop.run_until_complete(async_main())
    finally:
        loop.close()
    
    print("Depth-first summarization completed!")


if __name__ == "__main__":
    main()