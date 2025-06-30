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
from langchain.text_splitter import TokenTextSplitter
import traceback

# Configuration constants
MAX_CHUNKS = 8
MIN_SECTION_LENGTH = 50
DEFAULT_SECTION_MAX_TOKENS = 500

class PaperTree:
    """
    A hierarchical representation of a scientific paper that preserves structure
    for TreeRAG retrieval system.
    """
    
    def __init__(
        self, 
        title: str, 
        text: str, 
        abstract: Optional[str] = None, 
        parent: Optional['PaperTree'] = None, 
        section_max_tokens: int = DEFAULT_SECTION_MAX_TOKENS, 
        keep_splitting: bool = True
    ):
        self.title = title
        self.abstract = abstract
        self.text = text
        self.section_max_tokens = section_max_tokens
        self.parent = parent
        self.sections: List['PaperTree'] = []
        
        if keep_splitting:
            self.sections = self._split_to_sections()
        
        # For leaf nodes without explicit abstract, use the text itself
        if len(self.sections) == 0 and self.abstract is None:
            self.abstract = self.text

    def __repr__(self) -> str:
        """String representation showing the tree structure"""
        return self._get_full_path() + "\n" + "".join(section._tree_str() for section in self.sections)

    def get_depth(self) -> int:
        """Get the depth of this node in the tree"""
        return 0 if self.parent is None else 1 + self.parent.get_depth()

    def _get_full_path(self) -> str:
        """Get the full path from root to this node"""
        path_parts = []
        current = self
        while current is not None:
            path_parts.append(current.title)
            current = current.parent
        return " --> ".join(reversed(path_parts))
    
    def _tree_str(self) -> str:
        """Generate tree visualization string"""
        indent = "--" * self.get_depth()
        result = f"{indent}> {self.title}\n"
        for section in self.sections:
            result += section._tree_str()
        return result

    def _split_to_sections(self) -> List['PaperTree']:
        """Split text into hierarchical sections"""
        text = self.text

        # Try to split by section headers at appropriate depth
        sections = self._split_by_headers(text)
        if sections:
            return sections

        # Extract figures, tables, and other elements
        text, element_sections = self._extract_elements(text)

        # Split remaining text into chunks
        text_sections = self._split_text_into_chunks(text)
        if len(text_sections) == 0 and len(element_sections) > 0:
            text_sections.append(self._create_child_section(f"Text", text, keep_splitting=False))
        
        return element_sections + text_sections

    def _split_by_headers(self, text: str) -> List['PaperTree']:
        """Split text by LaTeX section headers"""
        depth = self.get_depth()

        if depth == 0 or self.title in "appendix":
            # At root level, match either sections or appendix. Also, appendix can contain \section{...}
            section_pattern = r"(\\section[\*\s]*(?:\[[^\]]*\])?\s*({(?:[^{}]*+|(?2))*}))"
            appendix_pattern = r"\\(appendix)\s+"
            pattern = f"{section_pattern}|{appendix_pattern}"
        else:
            # At deeper levels, match subsections
            pattern = r"(\\" + "sub" * depth + r"section[\*\s]*(?:\[[^\]]*\])?\s*({(?:[^{}]*+|(?2))*}))"

        matches = list(re.finditer(pattern, text))
        
        if not matches:
            return []

        sections = []
        start, title = 0, "header"
        
        for match in matches:
            end = match.start()
            if start < end:  # Only create section if there's content
                section_text = text[start:end]
                sections.append(self._create_child_section(title, section_text))
            
            start = match.end()
            title = match.group(3) if not match.group(2) else match.group(2)[1:-1]
            
            # Wait to split sections in the appendix
            if title in "appendix":
                break
        
        section_text = text[start:]
        sections.append(self._create_child_section(title, section_text))

        # There will be cases where we extract "header" and then "section1", but it makes more sense to just count this as a single section
        if len(sections) == 2 and sections[0].title in "header":
            section1 = sections[0]
            section2 = sections[1]
            if section1.sections:
                for section in section1.sections:
                    section.parent = self
                sections = section1.sections + [section2]
        
        return sections

    def _extract_elements(self, text: str) -> tuple[str, List['PaperTree']]:
        """Extract figures, tables, and other elements from text"""
        elements = {}
        cleaned_text = text
        
        for element_type in ["figure", "table", "sidewaystable"]:
            pattern = rf"\\begin\{{{element_type}\*?\}}(.*?)\\end\{{{element_type}\*?\}}"
            matches = re.findall(pattern, cleaned_text, re.DOTALL)
            elements[element_type] = matches
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.DOTALL)

        sections = []
        for element_type, element_list in elements.items():
            for i, element_content in enumerate(element_list):
                abstract = self._extract_caption(element_content, element_type)
                sections.append(PaperTree(
                    title=f"{element_type} {i}",
                    text=element_content,
                    abstract=abstract,
                    parent=self,
                    section_max_tokens=self.section_max_tokens,
                    keep_splitting=False,
                ))

        return cleaned_text, sections

    def _extract_caption(self, element_content: str, element_type: str) -> str:
        """Extract caption from figure/table content"""
        caption = ""
        
        # Standard caption patterns
        for caption_type in ["caption", "tbl", "tabcaption"]:
            pattern = rf"\\{caption_type}[\*\s]*(?:\[[^\]]*\])?\s*({{\s*(?:[^{{}}]*+|(?1))*\s*}})"
            matches = re.findall(pattern, element_content, re.DOTALL)
            caption += "".join(matches)
        
        # captionof pattern
        pattern = rf"\\captionof[\*\s]*{{\s*{element_type}\s*}}\s*({{\s*(?:[^{{}}]*+|(?1))*\s*}})"
        matches = re.findall(pattern, element_content, re.DOTALL)
        caption += "".join(matches)
        
        return caption.strip()

    def _split_text_into_chunks(self, text: str) -> List['PaperTree']:
        """Split text into appropriately sized chunks"""
        if len(re.sub(r"\s+", "", text)) < MIN_SECTION_LENGTH:
            return []

        text_splitter = TokenTextSplitter(
            chunk_size=self.section_max_tokens,
            chunk_overlap=self.section_max_tokens // 8,
            strip_whitespace=True,
        )
        chunks = text_splitter.split_text(text)
        
        if len(chunks) == 1:
            return []

        # If too many chunks, create subsections
        if len(chunks) > MAX_CHUNKS:
            return self._create_subsections(text, chunks)
        
        # Create individual chunk sections
        return [
            self._create_child_section(f"Chunk {i}", chunk, keep_splitting=False)
            for i, chunk in enumerate(chunks)
        ]

    def _create_subsections(self, text: str, chunks: List[str]) -> List['PaperTree']:
        """Create subsections when there are too many chunks"""
        num_sections = min(math.ceil(len(chunks) / (MAX_CHUNKS - 1)), MAX_CHUNKS)
        step = math.ceil(len(chunks) / num_sections)
        
        sections = []
        start_index = 0
        
        for i in range(0, len(chunks), step):
            end_chunk_idx = min(i + step, len(chunks))
            
            if end_chunk_idx >= len(chunks):
                section_text = text[start_index:]
            else:
                end_index = text.find(chunks[end_chunk_idx])
                section_text = text[start_index:end_index] if end_index != -1 else text[start_index:]
            
            sections.append(self._create_child_section(f"Subsection {len(sections)}", section_text))
            start_index = end_index if end_chunk_idx < len(chunks) else len(text)
        
        return sections

    def _create_child_section(self, title: str, text: str, keep_splitting: bool = True) -> 'PaperTree':
        """Helper to create a child section"""
        return PaperTree(
            title=title,
            text=text,
            abstract=None,
            parent=self,
            section_max_tokens=self.section_max_tokens,
            keep_splitting=keep_splitting,
        )

    def _split_to_sections_recursive(self, text: str) -> List['PaperTree']:
        """Recursively split text (used for appendix handling)"""
        temp_paper = PaperTree(
            title="temp",
            text=text,
            parent=self,
            section_max_tokens=self.section_max_tokens
        )
        return temp_paper.sections