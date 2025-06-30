import regex as re
import csv
import signal
import time
import os
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from tqdm import tqdm
from functools import partial
from pathlib import Path
import expand_latex_macros

def remove_headers(tex):
    """
    Remove all content before the abstract or titlepage
    """
    substrings = ["\\maketitle", "\\end{titlepage}", "\\end{abstract}", "\\abstract", "\\begin{document}"]
    max_index = 0
    for substring in substrings:
        index = tex.rfind(substring) + len(substring)
        if index > max_index:
            max_index = index
    return tex[max_index:]

def remove_boilerplate(tex):
    """
    Remove all LaTeX boilerplate code, comments, and bibliographies
    """
    patterns = [
        r"\\newpage", r"\\cleardoublepage", r"\\pagestyle\{[\w\d]+\}",  r"\\setcounter\{[\w\d]+\}\{\d+\}", 
        r"\\pagenumbering\{[\w\d]+\}", r"\\bibliographystyle\{[\w\d]+\}", r"\\end\{document\}", r"\\bibliography",
    ]
    for pattern in patterns:
        tex = re.sub(pattern, "", tex)

    # Remove all macros
    pattern = r"\\def\s*\\(\w+)\s*((?:#\d\s*)*)\s*({(?:[^{}]*+|(?3))*})"
    tex = re.sub(pattern, "", tex)
    pattern = r"\\newcommand\*?\s*{?\s*\\(\w+)\s*}?\s*((?:\[\s*\d+\s*\])*)\s*({(?:[^{}]*+|(?3))*})"
    tex = re.sub(pattern, "", tex)
    pattern = r"\\renewcommand\*?\s*{?\s*\\(\w+)\s*}?\s*((?:\[\s*\d+\s*\])*)\s*({(?:[^{}]*+|(?3))*})"
    tex = re.sub(pattern, "", tex)
    
    # Remove all comments
    pattern = r"\\begin\s*\{\s*comment\s*\}(.*?)\\end\s*\{\s*comment\s*\}"
    tex = re.sub(pattern, "", tex, flags=re.DOTALL)
    pattern = r"(?<!\\)%.*"
    tex = re.sub(pattern, "", tex)

    # Get rid of any bibliography
    pattern = r"\\bibitem\{.+\}(?:.|\n)*\\EndOfBibitem"
    tex = re.sub(pattern, "", tex)
    pattern = r"\\begin{thebibliography}(?:\n|.)*\\end{thebibliography}"
    tex = re.sub(pattern, "", tex)

    # Get rid of extra junk commands
    pattern = r"\\noindent"
    tex = re.sub(pattern, "", tex)
    pattern = r"\\bigskip"
    tex = re.sub(pattern, "", tex)
    pattern = r"\\mbox\{~\}"
    tex = re.sub(pattern, "", tex)
    pattern = r"\\clearpage"
    tex = re.sub(pattern, "", tex)
    pattern = r"\\twocolumn"
    tex = re.sub(pattern, "", tex)
    pattern = r"\\onecolumn"
    tex = re.sub(pattern, "", tex)
    pattern = r"\\tableofcontents"
    tex = re.sub(pattern, "", tex)
    
    return tex

def remove_lhcb_content(tex):
    """
    Remove all LHCb collab sections, lists of universities, etc
    """
    # LHCb junk
    pattern = r"\\centerline[\n\s]*\{[\n\s]*\\large[\n\s]*\\bf[\n\s]*LHCb[\n\s]*collaboration[\n\s]*\}[\n\s]*\\begin[\n\s]*\{[\n\s]*flushleft[\n\s]*\}(?:\n|.)*\{[\n\s]*\\footnotesize(?:\n|.)*\}[\n\s]*\\end[\n\s]*\{[\n\s]*flushleft[\n\s]*\}"
    tex = re.sub(pattern, "", tex)
    pattern = r"[a-zA-Z.-]+(?:~[a-zA-Z-\\ \{\}\"\'\`]*)+\$\^\{[a-zA-Z0-9,]+\}\$[\,.][\s\n]*"
    tex = re.sub(pattern, "", tex)
    pattern = r"\$\s*\^\{[\w\d\s]+\}\$.*\\"
    tex = re.sub(pattern, "", tex)
    pattern = r"\\begin\s*{\s*flushleft\s*}.*?\\end\s*{\s*flushleft\s*}"
    tex = re.sub(pattern, "", tex, flags=re.DOTALL)
    pattern = r'\\centerline\s*\{\s*(\\[a-zA-Z]+\s*)+.*\}'
    tex = re.sub(pattern, "", tex)
    return tex

def remove_section_content(tex, section_names):
    """
    Remove all content between a section command containing section_name and the next section command
    or end of document.
    """
    # Create a pattern that matches any section command followed by the section name
    # This handles \section, \section*, \subsection, etc.
    section_pattern = r'\\(?:sub)*section\*?{([^}]*)}'
    
    # Pattern for page breaks
    page_break_pattern = r'\\(?:new|clear)page'
    
    # Find all sections in the document
    sections = list(re.finditer(section_pattern, tex))
    
    # Process regular sections
    regions_to_remove = []
    for i, match in enumerate(sections):
        section_name = match.group(1)
        
        if any(name.lower() in section_name.lower() for name in section_names):
            start_pos = match.start()
            
            # Find the earliest ending point
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
    
    # Remove all regions in reverse order
    cleaned_content = tex
    for start, end in sorted(regions_to_remove, reverse=True):
        cleaned_content = cleaned_content[:start] + cleaned_content[end:]
    return cleaned_content.strip()

# Removes all formatting commands from a LaTeX string
def remove_double_brackets(input_string):
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
            match = re.match("\s*\}", remaining_str[first_closing_bracket+1:])
            if match:
                second_closing_bracket = first_closing_bracket + match.span()[1]
                new_str += remaining_str[:first_open_bracket+1] 
                remaining_str = remaining_str[second_open_bracket:first_closing_bracket] + remaining_str[second_closing_bracket:]
                changed = True
            else:
                new_str += remaining_str[:first_open_bracket+1]
                remaining_str = remaining_str[first_open_bracket+1:]
            search = re.search(r"\{\s*\{", remaining_str)
        new_str += remaining_str
        input_string = new_str
    return input_string

def clean_latex_spacing(text):

    # Replace linebreak \\ with newline
    text = re.sub(r"\s+\\\\(\[\s*[^\]]*\s*\])?\s+", "\n", text)
    # Remove lone \
    text = re.sub(r"\s+\\\s+", " ", text)

    # Pattern to match \mskip, \hskip, \vskip, etc. with their parameters
    spacing_pattern = r'\\[hvmtb]?skip\s+(-?\d*\.?\d*)(mu|pt|in|cm|mm|em|ex)?'
    # Pattern for more general spacing commands like \, \; \: \! \quad \qquad
    other_spacing = r'\\[,;:!]|\\(quad|qquad)\b'
    # Pattern for space \␣ (backslash followed by space)
    backslash_space = r'\\\s'
    # Pattern for phantom spacing commands
    phantom_pattern = r'\\[hv]?phantom(\{(?:[^{}]|(?1))*\})'
    
    # First remove the spacing commands with parameters
    text = re.sub(spacing_pattern, ' ', text)
    # Then remove the other spacing commands
    text = re.sub(other_spacing, ' ', text)
    # Remove backslash space
    text = re.sub(backslash_space, ' ', text)
    # Remove phantom spacing commands
    text = re.sub(phantom_pattern, ' ', text)
    
    return text

def improve_latex_symbol_consistency(tex):

    upgreek_to_greek = {
        # Lowercase letters
        "\\upalpha": "\\alpha",
        "\\upbeta": "\\beta",
        "\\upgamma": "\\gamma",
        "\\updelta": "\\delta",
        "\\upepsilon": "\\epsilon",
        "\\upvarepsilon": "\\varepsilon",
        "\\upzeta": "\\zeta",
        "\\upeta": "\\eta",
        "\\uptheta": "\\theta",
        "\\upvartheta": "\\vartheta",
        "\\upiota": "\\iota",
        "\\upkappa": "\\kappa",
        "\\uplambda": "\\lambda",
        "\\upmu": "\\mu",
        "\\upnu": "\\nu",
        "\\upxi": "\\xi",
        "\\upomicron": "\\omicron",
        "\\uppi": "\\pi",
        "\\upvarpi": "\\varpi",
        "\\uprho": "\\rho",
        "\\upvarrho": "\\varrho",
        "\\upsigma": "\\sigma",
        "\\upvarsigma": "\\varsigma",
        "\\uptau": "\\tau",
        "\\upupsilon": "\\upsilon",
        "\\upphi": "\\phi",
        "\\upvarphi": "\\varphi",
        "\\upchi": "\\chi",
        "\\uppsi": "\\psi",
        "\\upomega": "\\omega",
        
        # Uppercase letters
        "\\upGamma": "\\Gamma",
        "\\upDelta": "\\Delta",
        "\\upTheta": "\\Theta",
        "\\upLambda": "\\Lambda",
        "\\upXi": "\\Xi",
        "\\upPi": "\\Pi",
        "\\upSigma": "\\Sigma",
        "\\upUpsilon": "\\Upsilon",
        "\\upPhi": "\\Phi",
        "\\upPsi": "\\Psi",
        "\\upOmega": "\\Omega"
    }

    for upgreek in upgreek_to_greek:
        greek = upgreek_to_greek[upgreek].replace("\\", "\\\\")
        upgreek = upgreek.replace("\\", "\\\\")
        tex = re.sub(upgreek, greek, tex)
    
    tex = re.sub(r"\\rightarrow", r"\\to", tex)
    tex = re.sub(r"\\mathcal", r"\\cal", tex)

    tex = re.sub(r"\\rangle", ">", tex)
    tex = re.sub(r"\\langle", "<", tex)
    tex = re.sub(r"\\prime", "'", tex)
    tex = re.sub(r"\\colon", ":", tex)

    return tex

def get_command_mappings(tex_path, timeout_seconds=30):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Execution timed out while processing {tex_path}")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        tex = open(tex_path).read()
        return expand_latex_macros.get_command_mappings(tex)
    
    except TimeoutError as e:
        print(e)
        return {}
    
    finally:
        signal.alarm(0)

def clean_and_save_tex_file(tex_path, cleaned_tex_dir, command_mappings, sections_to_remove, timeout_seconds=30):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Execution timed out while processing {tex_path}")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        tex = open(tex_path).read()
        tex = remove_headers(tex)
        tex = remove_boilerplate(tex)
        tex = remove_lhcb_content(tex)
        tex = remove_section_content(tex, sections_to_remove)
        tex = expand_latex_macros.sub_macros_for_defs(tex, command_mappings)
        tex = clean_latex_spacing(tex)
        tex = remove_double_brackets(tex)
        tex = improve_latex_symbol_consistency(tex)
        # Remove excessive newlines, spaces, and all math mode declarations $
        tex = re.sub(r"\n[\s]+", "\n", tex)
        tex = re.sub(r"[ \t\r\f]+", " ", tex)
        tex = re.sub(r'(?<!\\)\$', '', tex)

        with open(cleaned_tex_dir / tex_path.name, "w") as file:
            file.write(tex)
    
    except TimeoutError as e:
        print(e)
    
    finally:
        signal.alarm(0)

def clean_and_expand_macros(tex_dir, cleaned_tex_dir, sections_to_remove=[]):
    n_cores = max(os.cpu_count() - 1, 1)
    headers = os.listdir(tex_dir)
    tex_paths = [Path(tex_dir) / header for header in filter(lambda str: ".tex" in str, headers)]

    command_mappings_all = []
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = list(tqdm(executor.map(get_command_mappings, tex_paths), 
                            total=len(tex_paths), 
                            desc=f"Parsing & cleaning LaTeX Macros from {tex_dir}"))
        command_mappings_all.extend(results)
    command_mappings = dict(chain.from_iterable(d.items() for d in command_mappings_all))

    curried_clean_and_save_tex_file = partial(clean_and_save_tex_file, cleaned_tex_dir=Path(cleaned_tex_dir), command_mappings=command_mappings, sections_to_remove=sections_to_remove)
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = list(tqdm(executor.map(curried_clean_and_save_tex_file, tex_paths), 
                            total=len(tex_paths), 
                            desc=f"Cleaning tex files and saving to {cleaned_tex_dir}"))
        
    return command_mappings

if __name__ == '__main__':
    tex_dir = "/work/submit/mcgreivy/beauty-in-stats/src/scraper/data/expanded_tex/"
    cleaned_tex_dir = "/work/submit/mcgreivy/beauty-in-stats/src/scraper/data/cleaned_tex/"
    sections_to_remove = [
        'Acknowledgements',
        'Acknowledgments',
        'References',
        'Bibliography',
    ]
    os.makedirs(cleaned_tex_dir, exist_ok = True)
    clean_and_expand_macros(tex_dir, cleaned_tex_dir, sections_to_remove)