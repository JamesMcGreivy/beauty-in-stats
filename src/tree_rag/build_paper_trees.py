"""
Author: James McGreivy
Email: mcgreivy@mit.edu
"""

import os
import regex as re
import math
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from pathlib import Path
from typing import List, Optional
import traceback

from PaperTree import PaperTree

class PaperTreeProcessor:
    """Handles the processing and cleanup of paper trees"""
    
    @staticmethod
    def remove_empty_sections(paper: PaperTree) -> None:
        """Remove sections with insufficient content"""
        paper.sections = [
            section for section in paper.sections
            if len(re.sub(r"\s", "", section.text)) >= PaperTree.MIN_SECTION_LENGTH
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


def process_single_file(filename: str, dir_tex: str, dir_abstracts: str) -> List[PaperTree]:
    """Process a single LaTeX file into a PaperTree"""
    try:
        # Load abstract if available
        abstract_path = Path(dir_abstracts) / filename
        try:
            with open(abstract_path, 'r', encoding='utf-8') as f:
                abstract = f.read().strip()
        except FileNotFoundError:
            print(f"No abstract found for {filename}")
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
    DEFAULT_MAX_WORKERS = 30
    dir_tex = "../scraper/data/cleaned_tex"
    dir_abstracts = "../scraper/data/abstracts"
    dir_paper_trees = "./paper_trees_cache/summarized_paper_trees/"
    
    # Ensure output directory exists
    os.makedirs(dir_paper_trees, exist_ok=True)
    
    # Get all LaTeX files
    tex_files = [f for f in os.listdir(dir_tex) if f.endswith('.tex')]
    print(f"Found {len(tex_files)} LaTeX files to process")
    
    # Process files in parallel
    papers = []
    with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_single_file, filename, dir_tex, dir_abstracts)
            for filename in tex_files
        ]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing papers"):
            papers.extend(future.result())
    
    print(f"Successfully processed {len(papers)} papers")
    
    # Handle missing captions
    missing_captions = PaperTreeProcessor.find_missing_captions(papers)
    if missing_captions:
        handle_missing_captions(missing_captions)
    
    # Save all papers
    print("Saving processed papers...")
    for paper in tqdm(papers, desc="Saving"):
        output_filename = paper.title.replace('.tex', '.pkl')
        output_path = Path(dir_paper_trees) / output_filename
        
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(paper, f)
        except Exception as e:
            print(f"Failed to save {output_filename}: {e}")
    
    print(f"Processing complete! Saved {len(papers)} papers to {dir_paper_trees}")


if __name__ == "__main__":
    main()