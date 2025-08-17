"""
Author: James McGreivy
Email: mcgreivy@mit.edu
"""

import os
import regex as re
import math
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
import traceback

from rag.tree.core.paper_tree import PaperTree
import rag.settings as settings
from rag.tree.core.utils import save_paper_trees, check_paper_exists, generate_filename

class PaperTreeProcessor:
    """Handles the processing and cleanup of paper trees"""
    
    @staticmethod
    def remove_empty_sections(paper: PaperTree) -> None:
        """Remove sections with insufficient content"""
        paper.sections = [
            section for section in paper.sections
            if len(re.sub(r"\s", "", section.text)) >= settings.PAPER_TREE_CONFIG["min_section_length"]
        ]
        
        for section in paper.sections:
            PaperTreeProcessor.remove_empty_sections(section)

    @staticmethod
    def collapse_single_subsections(paper: PaperTree) -> None:
        """Collapse sections that have only one subsection"""
        new_sections = []
        
        for section in paper.sections:
            if len(section.sections) == 1:
                subsection = section.sections[0]
                subsection.parent = paper
                subsection.title = f"{section.title} - {subsection.title}"
                new_sections.append(subsection)
            else:
                new_sections.append(section)
        
        paper.sections = new_sections
        
        # Recursively apply to all sections
        for section in paper.sections:
            PaperTreeProcessor.collapse_single_subsections(section)

    @staticmethod
    def find_missing_captions(papers: List[PaperTree]) -> List[PaperTree]:
        """Find elements that need manual caption extraction"""
        missing_captions = []
        
        def collect_missing(paper: PaperTree) -> None:
            for section in paper.sections:
                collect_missing(section)
            
            if (paper.abstract is not None and 
                any(element in paper.title for element in ["figure", "table", "sidewaystable"]) and
                len(paper.abstract.strip()) == 0):
                missing_captions.append(paper)
        
        for paper in papers:
            collect_missing(paper)
        
        return missing_captions


class LaTeXCleaner:
    """Handles cleaning of LaTeX source files"""
    
    @staticmethod
    def clean_latex(tex: str) -> str:
        """Clean LaTeX source of unnecessary content"""
        # Find content start after title/abstract
        tex = LaTeXCleaner._find_content_start(tex)
        
        # Remove various LaTeX commands
        tex = LaTeXCleaner._remove_latex_commands(tex)
        
        # Remove macros and custom commands
        tex = LaTeXCleaner._remove_macros(tex)
        
        # Remove comments
        tex = LaTeXCleaner._remove_comments(tex)
        
        # Remove bibliography
        tex = LaTeXCleaner._remove_bibliography(tex)
        
        # Remove LHCb-specific content
        tex = LaTeXCleaner._remove_lhcb_content(tex)
        
        # Clean up whitespace
        tex = re.sub(r"\s{2,}", " ", tex)
        
        return tex.strip()

    @staticmethod
    def _find_content_start(tex: str) -> str:
        """Find where the main content starts"""
        markers = ["\\maketitle", "\\end{titlepage}", "\\end{abstract}", "\\abstract"]
        max_index = 0
        
        for marker in markers:
            index = tex.rfind(marker)
            if index != -1:
                max_index = max(max_index, index + len(marker))
        
        return tex[max_index:]

    @staticmethod
    def _remove_latex_commands(tex: str) -> str:
        """Remove standard LaTeX commands"""
        patterns = [
            r"\\newpage", r"\\cleardoublepage", r"\\pagestyle\{[\w\d]+\}",
            r"\\setcounter\{[\w\d]+\}\{\d+\}", r"\\pagenumbering\{[\w\d]+\}",
            r"\\bibliographystyle\{[\w\d]+\}", r"\\end\{document\}", r"\\bibliography",
        ]
        
        for pattern in patterns:
            tex = re.sub(pattern, "", tex)
        
        return tex

    @staticmethod
    def _remove_macros(tex: str) -> str:
        """Remove macro definitions"""
        patterns = [
            r"\\def\s*\\(\w+)\s*((?:#\d\s*)*)\s*({(?:[^{}]*+|(?3))*})",
            r"\\newcommand\*?\s*{?\s*\\(\w+)\s*}?\s*((?:\[\s*\d+\s*\])*)\s*({(?:[^{}]*+|(?3))*})",
            r"\\renewcommand\*?\s*{?\s*\\(\w+)\s*}?\s*((?:\[\s*\d+\s*\])*)\s*({(?:[^{}]*+|(?3))*})",
        ]
        
        for pattern in patterns:
            tex = re.sub(pattern, "", tex)
        
        return tex

    @staticmethod
    def _remove_comments(tex: str) -> str:
        """Remove LaTeX comments"""
        # Remove comment blocks
        tex = re.sub(r"\\begin\s*\{\s*comment\s*\}(.*?)\\end\s*\{\s*comment\s*\}", "", tex, flags=re.DOTALL)
        # Remove line comments
        tex = re.sub(r"(?<!\\)%.*", "", tex)
        return tex

    @staticmethod
    def _remove_bibliography(tex: str) -> str:
        """Remove bibliography sections"""
        patterns = [
            r"\\bibitem\{.+\}(?:.|\n)*\\EndOfBibitem",
            r"\\begin{thebibliography}(?:\n|.)*\\end{thebibliography}",
        ]
        
        for pattern in patterns:
            tex = re.sub(pattern, "", tex, flags=re.DOTALL)
        
        return tex

    @staticmethod
    def _remove_lhcb_content(tex: str) -> str:
        """Remove LHCb-specific content"""
        patterns = [
            r"\\centerline[\n\s]*\{[\n\s]*\\large[\n\s]*\\bf[\n\s]*LHCb[\n\s]*collaboration[\n\s]*\}[\n\s]*\\begin[\n\s]*\{[\n\s]*flushleft[\n\s]*\}(?:\n|.)*\{[\n\s]*\\footnotesize(?:\n|.)*\}[\n\s]*\\end[\n\s]*\{[\n\s]*flushleft[\n\s]*\}",
            r"[a-zA-Z.-]+(?:~[a-zA-Z-\\ \{\}\"\'\`]*)+\$\^\{[a-zA-Z0-9,]+\}\$[\,.][\s\n]*",
            r"\$\s*\^\{[\w\d\s]+\}\$.*\\",
        ]
        
        for pattern in patterns:
            tex = re.sub(pattern, "", tex, flags=re.DOTALL)
        
        return tex


def should_process_file(tex_filename: str, output_dir: str) -> bool:
    """
    Determine if a file should be processed based on cache existence and configuration.
    
    Args:
        tex_filename: Name of the .tex file to check
        output_dir: Directory where cached .pkl files are stored
        
    Returns:
        True if the file should be processed, False if it can be skipped
    """
    # Check if rebuild is forced in configuration
    if settings.PAPER_TREE_CONFIG.get("overwrite_existing", False):
        return True
    
    # Use utils function to check if paper already exists
    return not check_paper_exists(tex_filename, output_dir)


def filter_files_to_process(tex_files: List[str], output_dir: str) -> List[str]:
    """
    Filter the list of tex files to only include those that need processing.
    
    Args:
        tex_files: List of .tex filenames to consider
        output_dir: Directory where cached .pkl files are stored
        
    Returns:
        Filtered list of .tex filenames that need processing
    """
    files_to_process = []
    skipped_count = 0
    
    for tex_file in tex_files:
        if should_process_file(tex_file, output_dir):
            files_to_process.append(tex_file)
        else:
            skipped_count += 1
    
    if skipped_count > 0:
        print(f"Skipping {skipped_count} files that already exist in cache")
        overwrite_setting = settings.PAPER_TREE_CONFIG.get("overwrite_existing", False)
        print(f"Set PAPER_TREE_CONFIG['overwrite_existing'] = True in settings to overwrite the cache and force a rebuild (currently: {overwrite_setting})")
    
    return files_to_process


def process_single_file(filename: str, dir_tex: str, dir_abstracts: str) -> List[PaperTree]:
    """Process a single LaTeX file into a PaperTree"""
    try:
        # Load abstract if available
        abstract_path = Path(dir_abstracts) / filename
        try:
            with open(abstract_path, 'r', encoding='utf-8') as f:
                abstract = f.read().strip()
        except FileNotFoundError:
            # Don't print for every missing abstract - too verbose
            abstract = None
        except Exception as e:
            print(f"Failed to load abstract for {filename}: {e}")
            abstract = None

        # Load and clean LaTeX content
        tex_path = Path(dir_tex) / filename
        with open(tex_path, 'r', encoding='utf-8') as f:
            full_tex = f.read()
        
        # Clean the LaTeX
        cleaned_tex = LaTeXCleaner.clean_latex(full_tex)
        
        # Create paper tree
        paper = PaperTree(
            filename, 
            cleaned_tex, 
            abstract=abstract,
        )
        
        # Apply cleanup procedures
        PaperTreeProcessor.remove_empty_sections(paper)
        PaperTreeProcessor.collapse_single_subsections(paper)
        
        return [paper]
        
    except Exception as e:
        print(f"Failed to process {filename}: {e}")
        traceback.print_exc()
        return []


def handle_missing_captions(missing_elements: List[PaperTree]) -> None:
    """Interactively handle elements with missing captions"""
    if not missing_elements:
        print("All elements have captions!")
        return
    
    print(f"Found {len(missing_elements)} elements with missing captions.")
    
    for element in missing_elements:
        print(f"\nElement: {element._get_full_path()}")
        print(f"Text preview: {element.text[:200]}...")
        print(f"Current caption: '{element.abstract}'")
        
        caption = input("Enter caption (or 'skip' to continue, 'quit' to stop): ").strip()
        
        if caption.lower() == 'quit':
            break
        elif caption.lower() != 'skip':
            element.abstract = caption


def main():
    """Main processing function"""
    # Configuration
    DEFAULT_MAX_WORKERS = settings.MAX_WORKERS
    dir_tex = settings.CLEAN_TEX_DIR
    dir_abstracts = settings.ABSTRACT_DIR
    output_dir = str(settings.TREE_RAG_CACHE)  # Convert to string for utils functions
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all LaTeX files
    all_tex_files = [f for f in os.listdir(dir_tex) if f.endswith('.tex')]
    print(f"Found {len(all_tex_files)} LaTeX files")
    
    # Filter files that need processing based on cache and configuration
    tex_files_to_process = filter_files_to_process(all_tex_files, output_dir)
    
    if not tex_files_to_process:
        print("No files need processing. All paper trees already exist in cache.")
        print("Set PAPER_TREE_CONFIG['overwrite_existing'] = True to force rebuild.")
        return
    
    print(f"Processing {len(tex_files_to_process)} files")
    
    # Process files in parallel
    papers = []
    with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_single_file, filename, dir_tex, dir_abstracts)
            for filename in tex_files_to_process
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing papers"):
            papers.extend(future.result())
    
    print(f"Successfully processed {len(papers)} papers")
    
    # Handle missing captions
    missing_captions = PaperTreeProcessor.find_missing_captions(papers)
    if missing_captions:
        handle_missing_captions(missing_captions)
    
    # Save all papers using utils function
    print("Saving processed papers...")
    saved_count = save_paper_trees(papers, output_dir, show_progress=True)
    
    print(f"Processing complete! Saved {saved_count} papers to {output_dir}")


if __name__ == "__main__":
    main()