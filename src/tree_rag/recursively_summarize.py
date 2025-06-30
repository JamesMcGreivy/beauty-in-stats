"""
Author: James McGreivy
Email: mcgreivy@mit.edu
"""

import regex as re
from tqdm import tqdm
import pickle
import os
import asyncio
import time
from pathlib import Path

from PaperTree import PaperTree

from llama_index.llms.openai import OpenAI
llm = OpenAI(temperature=0, model="gpt-4.1-nano")

# Rate limiting configuration
SEMAPHORE = asyncio.Semaphore(3)  # Reduced for safety
REQUEST_DELAY = 2.0  # Seconds between requests
BATCH_SIZE = 50

def get_prompt(excerpt, section_level=None, section_title=None):
    prompt = f"""
You are creating a hierarchical summary for a TreeRAG system that will enable precise scientific literature search. This summary will be used for semantic similarity matching during tree traversal from paper root to relevant leaf content.

CONTEXT: This is a summary of a paper section that contains multiple subsections/content units. During retrieval, researchers will:
1. Start at paper root and traverse down the hierarchy
2. Use your summary to decide if this section branch is relevant to their query
3. Continue down to leaf nodes if this section seems topically relevant

CRITICAL REQUIREMENTS:
- Topical Scope: Clearly define what topics/themes this section covers so retrieval can determine relevance
- Technical Precision: Preserve all domain-specific terminology, particle names, experimental methods, statistical approaches
- Hierarchical Context: Focus on what makes this section distinct from sibling sections
- Query-Friendly: Write in language that matches how physicists would formulate search queries

WHAT TO INCLUDE:
- Primary research questions or objectives addressed in this section
- Key experimental methods, theoretical frameworks, or analysis techniques used
- Main findings, measurements, or conclusions
- Important particle physics concepts, detector components, or statistical methods
- Distinctive technical details that differentiate this content

WHAT TO EXCLUDE:
- Generic LHC/LHCb background (unless specifically relevant to this section's focus)
- Basic particle physics definitions (unless central to this section's contribution)
- Citation numbers and reference lists
- Mathematical notation (describe the physics/meaning instead)

FORMAT: Write 2-3 focused paragraphs (~400 tokens total) that capture the essence of what a physicist would find in this section.

{f"Additional Section Context: {section_title}" if section_title else ""}

### Content to Summarize:
{excerpt}

### Summary:
"""
    return prompt

async def summarize_with_retry(paper, max_retries=3):
    """Summarize with exponential backoff retry logic"""
    async with SEMAPHORE:
        await asyncio.sleep(REQUEST_DELAY)
        
        text = ""
        for section in paper.sections:
            elements = ["figure", "table", "sidewaystable"]
            if any([element in section.title for element in elements]):
                text += f"{section.title} caption: {section.abstract} \n"
            else:
                text += f"{section.abstract} \n"
        
        # Get section title for context
        section_title = getattr(paper, 'title', None)
        
        for attempt in range(max_retries):
            try:
                result = await llm.acomplete(get_prompt(text, section_title=section_title))
                return result
            except Exception as e:
                wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
                print(f"Attempt {attempt + 1} failed for paper {section_title}: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Failed to summarize after {max_retries} attempts")
                    raise e

async def process_batch(to_be_summarized):
    """Process a batch of papers with proper error handling"""
    print(f"Processing batch of {len(to_be_summarized)} papers...")
    
    tasks = [summarize_with_retry(paper) for paper in to_be_summarized]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_count = 0
    for paper, result in zip(to_be_summarized, results):
        if isinstance(result, Exception):
            print(f"Failed to summarize paper {getattr(paper, 'title', 'Unknown')}: {result}")
            continue
        paper.abstract = result.text
        successful_count += 1
    
    print(f"Successfully processed {successful_count}/{len(to_be_summarized)} papers")

def build_to_be_summarized(paper, to_be_summarized):
    """Build list of papers that need summarization"""
    if paper.abstract is None:
        can_be_summarized = True
        for section in paper.sections:
            if section.abstract is None:
                can_be_summarized = False
                break
        
        if can_be_summarized:
            to_be_summarized.append(paper)

    for section in paper.sections:
        build_to_be_summarized(section, to_be_summarized)

def give_leaf_abstracts(papers):
    """Set abstracts for leaf nodes (text content)"""
    def give_leaf_abstract(paper):
        if paper.abstract is None and len(paper.sections) == 0:
            print("hmmmmm")
            paper.abstract = paper.text
        for section in paper.sections:
            give_leaf_abstract(section)
    
    for paper in papers:
        give_leaf_abstract(paper)

def save_papers(papers, output_dir):
    """Save all papers to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    for paper in papers:
        filename = paper.title[:-4] + ".pkl" if paper.title.endswith('.pkl') else paper.title + ".pkl"
        filepath = Path(output_dir) / filename
        with open(filepath, "wb") as f:
            pickle.dump(paper, f)

async def main():
    # Directory paths
    load_dir = "./paper_trees_cache/summarized_paper_trees/"
    output_dir = "./paper_trees_cache/summarized_paper_trees/"
    
    # Load all paper trees
    print("Loading paper trees...")
    filenames = [f for f in os.listdir(load_dir) if f.endswith('.pkl')]
    papers = []
    
    for filename in tqdm(filenames, desc="Loading papers"):
        try:
            with open(os.path.join(load_dir, filename), "rb") as f:
                paper = pickle.load(f)
                papers.append(paper)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
    
    print(f"Loaded {len(papers)} papers")
    
    # Initialize leaf abstracts
    give_leaf_abstracts(papers)
    
    # Process papers level by level until all are summarized
    iteration = 0
    while True:
        iteration += 1
        to_be_summarized = []
        
        for paper in papers:
            build_to_be_summarized(paper, to_be_summarized)
        
        if len(to_be_summarized) == 0:
            print("All papers fully summarized!")
            break
            
        print(f"\nIteration {iteration}: {len(to_be_summarized)} sections need summarization")
        
        # Process in batches
        for i in range(0, len(to_be_summarized), BATCH_SIZE):
            batch = to_be_summarized[i:i + BATCH_SIZE]
            await process_batch(batch)
            
            # Save progress after each batch
            save_papers(papers, output_dir)
            print(f"Progress saved after batch {i//BATCH_SIZE + 1}")
    
    # Final save
    save_papers(papers, output_dir)
    print(f"All papers saved to {output_dir}")

if __name__ == "__main__":
    asyncio.run(main())