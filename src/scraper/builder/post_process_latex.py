"""LaTeX post-processing utilities for cleaning and standardizing academic papers."""

import os
import signal
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional

import regex as re
from tqdm import tqdm

import expand_latex_macros


def remove_headers(tex: str) -> str:
    """Remove all content before the abstract, titlepage, or main document."""
    substrings = [
        "\\maketitle", 
        "\\end{titlepage}", 
        "\\end{abstract}", 
        "\\abstract", 
        "\\begin{document}"
    ]
    max_index = 0
    for substring in substrings:
        index = tex.rfind(substring) + len(substring)
        if index > max_index:
            max_index = index
    return tex[max_index:]


def remove_boilerplate(tex: str) -> str:
    """Remove LaTeX boilerplate code, comments, and bibliographies."""
    # Remove common boilerplate commands
    patterns = [
        r"\\newpage", 
        r"\\cleardoublepage", 
        r"\\pagestyle\{[\w\d]+\}",  
        r"\\setcounter\{[\w\d]+\}\{\d+\}", 
        r"\\pagenumbering\{[\w\d]+\}", 
        r"\\bibliographystyle\{[\w\d]+\}", 
        r"\\end\{document\}", 
        r"\\bibliography",
    ]
    for pattern in patterns:
        tex = re.sub(pattern, "", tex)

    # Remove macro definitions
    macro_patterns = [
        r"\\def\s*\\(\w+)\s*((?:#\d\s*)*)\s*({(?:[^{}]*+|(?3))*})",
        r"\\newcommand\*?\s*{?\s*\\(\w+)\s*}?\s*((?:\[\s*\d+\s*\])*)\s*({(?:[^{}]*+|(?3))*})",
        r"\\renewcommand\*?\s*{?\s*\\(\w+)\s*}?\s*((?:\[\s*\d+\s*\])*)\s*({(?:[^{}]*+|(?3))*})"
    ]
    for pattern in macro_patterns:
        tex = re.sub(pattern, "", tex)
    
    # Remove comments
    tex = re.sub(r"\\begin\s*\{\s*comment\s*\}(.*?)\\end\s*\{\s*comment\s*\}", "", tex, flags=re.DOTALL)
    tex = re.sub(r"(?<!\\)%.*", "", tex)

    # Remove bibliography sections
    tex = re.sub(r"\\bibitem\{.+\}(?:.|\n)*\\EndOfBibitem", "", tex)
    tex = re.sub(r"\\begin{thebibliography}(?:\n|.)*\\end{thebibliography}", "", tex)

    # Remove formatting commands
    formatting_patterns = [
        r"\\noindent", r"\\bigskip", r"\\mbox\{~\}", r"\\clearpage",
        r"\\twocolumn", r"\\onecolumn", r"\\tableofcontents"
    ]
    for pattern in formatting_patterns:
        tex = re.sub(pattern, "", tex)
    
    return tex


def remove_lhcb_content(tex: str) -> str:
    """Remove LHCb collaboration sections and university lists."""
    # Remove LHCb collaboration header
    lhcb_header = (r"\\centerline[\n\s]*\{[\n\s]*\\large[\n\s]*\\bf[\n\s]*LHCb[\n\s]*"
                   r"collaboration[\n\s]*\}[\n\s]*\\begin[\n\s]*\{[\n\s]*flushleft[\n\s]*\}"
                   r"(?:\n|.)*\{[\n\s]*\\footnotesize(?:\n|.)*\}[\n\s]*\\end[\n\s]*\{[\n\s]*"
                   r"flushleft[\n\s]*\}")
    tex = re.sub(lhcb_header, "", tex)
    
    # Remove author affiliations and university references
    tex = re.sub(r"[a-zA-Z.-]+(?:~[a-zA-Z-\\ \{\}\"\'\`]*)+\$\^\{[a-zA-Z0-9,]+\}\$[\,.][\s\n]*", "", tex)
    tex = re.sub(r"\$\s*\^\{[\w\d\s]+\}\$.*\\", "", tex)
    tex = re.sub(r"\\begin\s*{\s*flushleft\s*}.*?\\end\s*{\s*flushleft\s*}", "", tex, flags=re.DOTALL)
    tex = re.sub(r'\\centerline\s*\{\s*(\\[a-zA-Z]+\s*)+.*\}', "", tex)
    
    return tex


def remove_section_content(tex: str, section_names: List[str]) -> str:
    """Remove content from specified sections until the next section or page break."""
    if not section_names:
        return tex
    
    section_pattern = r'\\(?:sub)*section\*?{([^}]*)}'
    page_break_pattern = r'\\(?:new|clear)page'
    
    # Find all sections in the document
    sections = list(re.finditer(section_pattern, tex))
    
    # Identify regions to remove
    regions_to_remove = []
    for i, match in enumerate(sections):
        section_name = match.group(1)
        
        if any(name.lower() in section_name.lower() for name in section_names):
            start_pos = match.start()
            end_pos = len(tex)
            
            # Check for next section
            if i < len(sections) - 1:
                end_pos = min(end_pos, sections[i + 1].start())
            
            # Check for page breaks
            page_break_match = re.search(page_break_pattern, tex[start_pos:])
            if page_break_match:
                possible_end = start_pos + page_break_match.start()
                end_pos = min(end_pos, possible_end)
                
            regions_to_remove.append((start_pos, end_pos))
    
    # Remove regions in reverse order to maintain indices
    cleaned_content = tex
    for start, end in sorted(regions_to_remove, reverse=True):
        cleaned_content = cleaned_content[:start] + cleaned_content[end:]
        
    return cleaned_content.strip()


def remove_double_brackets(input_string: str) -> str:
    """Remove redundant nested braces from LaTeX formatting commands."""
    changed = True
    while changed:
        changed = False
        new_str = ""
        remaining_str = input_string
        search = re.search(r"\{\s*\{", remaining_str)
        
        while search:
            first_open_bracket = search.span()[0]
            second_open_bracket = search.span()[1]
            first_closing_bracket = expand_latex_macros.find_matching_brace(remaining_str, second_open_bracket)
            match = re.match(r"\s*\}", remaining_str[first_closing_bracket+1:])
            
            if match:
                second_closing_bracket = first_closing_bracket + match.span()[1]
                new_str += remaining_str[:first_open_bracket+1] 
                remaining_str = (remaining_str[second_open_bracket:first_closing_bracket] + 
                               remaining_str[second_closing_bracket:])
                changed = True
            else:
                new_str += remaining_str[:first_open_bracket+1]
                remaining_str = remaining_str[first_open_bracket+1:]
                
            search = re.search(r"\{\s*\{", remaining_str)
            
        new_str += remaining_str
        input_string = new_str
        
    return input_string


def clean_latex_spacing(text: str) -> str:
    """Remove LaTeX spacing commands and normalize whitespace."""
    # Replace linebreak \\ with newline
    text = re.sub(r"\s+\\\\(\[\s*[^\]]*\s*\])?\s+", "\n", text)
    # Remove lone backslashes
    text = re.sub(r"\s+\\\s+", " ", text)

    # Remove various spacing commands
    spacing_patterns = [
        r'\\[hvmtb]?skip\s+(-?\d*\.?\d*)(mu|pt|in|cm|mm|em|ex)?',  # \skip commands
        r'\\[,;:!]|\\(quad|qquad)\b',  # spacing shortcuts
        r'\\\s',  # backslash space
        r'\\[hv]?phantom(\{(?:[^{}]|(?1))*\})'  # phantom commands
    ]
    
    for pattern in spacing_patterns:
        text = re.sub(pattern, ' ', text)
    
    return text


def improve_latex_symbol_consistency(tex: str) -> str:
    """Standardize LaTeX symbol commands for consistency."""
    # Map upright Greek letters to standard Greek letters
    upgreek_to_greek = {
        # Lowercase letters
        "\\upalpha": "\\alpha", "\\upbeta": "\\beta", "\\upgamma": "\\gamma",
        "\\updelta": "\\delta", "\\upepsilon": "\\epsilon", "\\upvarepsilon": "\\varepsilon",
        "\\upzeta": "\\zeta", "\\upeta": "\\eta", "\\uptheta": "\\theta",
        "\\upvartheta": "\\vartheta", "\\upiota": "\\iota", "\\upkappa": "\\kappa",
        "\\uplambda": "\\lambda", "\\upmu": "\\mu", "\\upnu": "\\nu",
        "\\upxi": "\\xi", "\\upomicron": "\\omicron", "\\uppi": "\\pi",
        "\\upvarpi": "\\varpi", "\\uprho": "\\rho", "\\upvarrho": "\\varrho",
        "\\upsigma": "\\sigma", "\\upvarsigma": "\\varsigma", "\\uptau": "\\tau",
        "\\upupsilon": "\\upsilon", "\\upphi": "\\phi", "\\upvarphi": "\\varphi",
        "\\upchi": "\\chi", "\\uppsi": "\\psi", "\\upomega": "\\omega",
        
        # Uppercase letters
        "\\upGamma": "\\Gamma", "\\upDelta": "\\Delta", "\\upTheta": "\\Theta",
        "\\upLambda": "\\Lambda", "\\upXi": "\\Xi", "\\upPi": "\\Pi",
        "\\upSigma": "\\Sigma", "\\upUpsilon": "\\Upsilon", "\\upPhi": "\\Phi",
        "\\upPsi": "\\Psi", "\\upOmega": "\\Omega"
    }

    # Apply Greek letter replacements
    for upgreek, greek in upgreek_to_greek.items():
        escaped_upgreek = upgreek.replace("\\", "\\\\")
        escaped_greek = greek.replace("\\", "\\\\")
        tex = re.sub(escaped_upgreek, escaped_greek, tex)
    
    # Standardize other symbols
    symbol_replacements = [
        (r"\\rightarrow", r"\\to"),
        (r"\\mathcal", r"\\cal"),
        (r"\\rangle", ">"),
        (r"\\langle", "<"),
        (r"\\prime", "'"),
        (r"\\colon", ":")
    ]
    
    for old_symbol, new_symbol in symbol_replacements:
        tex = re.sub(old_symbol, new_symbol, tex)

    return tex


def get_command_mappings(tex_path: Path, timeout_seconds: int = 30) -> Dict[str, str]:
    """Extract LaTeX command mappings from a file with timeout protection."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Execution timed out while processing {tex_path}")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        with open(tex_path, 'r', encoding='utf-8') as file:
            tex = file.read()
        return expand_latex_macros.get_command_mappings(tex)
    
    except TimeoutError as e:
        print(e)
        return {}
    except Exception as e:
        print(f"Error processing {tex_path}: {e}")
        return {}
    
    finally:
        signal.alarm(0)


def clean_and_save_tex_file(
    tex_path: Path, 
    cleaned_tex_dir: Path, 
    command_mappings: Dict[str, str], 
    sections_to_remove: List[str], 
    timeout_seconds: int = 30
) -> None:
    """Clean a single LaTeX file and save the result."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Execution timed out while processing {tex_path}")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        with open(tex_path, 'r', encoding='utf-8') as file:
            tex = file.read()
        
        # Apply cleaning pipeline
        tex = remove_headers(tex)
        tex = remove_boilerplate(tex)
        tex = remove_lhcb_content(tex)
        tex = remove_section_content(tex, sections_to_remove)
        tex = expand_latex_macros.sub_macros_for_defs(tex, command_mappings)
        tex = clean_latex_spacing(tex)
        tex = remove_double_brackets(tex)
        tex = improve_latex_symbol_consistency(tex)
        
        # Final cleanup: normalize whitespace and remove math mode delimiters
        tex = re.sub(r"\n[\s]+", "\n", tex)
        tex = re.sub(r"[ \t\r\f]+", " ", tex)
        tex = re.sub(r'(?<!\\)\$', '', tex)

        output_path = cleaned_tex_dir / tex_path.name
        with open(output_path, "w", encoding='utf-8') as file:
            file.write(tex)
    
    except TimeoutError as e:
        print(e)
    except Exception as e:
        print(f"Error processing {tex_path}: {e}")
    
    finally:
        signal.alarm(0)


def clean_and_expand_macros(
    tex_dir: Path, 
    cleaned_tex_dir: Path, 
    sections_to_remove: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Clean and process all LaTeX files in a directory using multiprocessing.
    
    Returns a dictionary of all command mappings found across all files.
    """
    if sections_to_remove is None:
        sections_to_remove = []
    
    # Ensure output directory exists
    cleaned_tex_dir.mkdir(parents=True, exist_ok=True)

    # Set up multiprocessing
    n_cores = max(os.cpu_count() - 1, 1)
    tex_files = [f for f in os.listdir(tex_dir) if f.endswith('.tex')]
    tex_paths = [tex_dir / filename for filename in tex_files]

    if not tex_paths:
        print(f"No .tex files found in {tex_dir}")
        return {}

    # First pass: extract command mappings from all files
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        command_mappings_list = list(
            tqdm(
                executor.map(get_command_mappings, tex_paths), 
                total=len(tex_paths), 
                desc=f"Parsing LaTeX macros from {tex_dir}"
            )
        )
    
    # Combine all command mappings
    command_mappings = dict(chain.from_iterable(d.items() for d in command_mappings_list))

    # Second pass: clean and save all files
    clean_func = partial(
        clean_and_save_tex_file, 
        cleaned_tex_dir=cleaned_tex_dir, 
        command_mappings=command_mappings, 
        sections_to_remove=sections_to_remove
    )
    
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        list(
            tqdm(
                executor.map(clean_func, tex_paths), 
                total=len(tex_paths), 
                desc=f"Cleaning files and saving to {cleaned_tex_dir}"
            )
        )
        
    return command_mappings