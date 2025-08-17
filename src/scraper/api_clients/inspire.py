from typing import Optional, List, Dict, Any
from pathlib import Path
import requests
import tarfile
import subprocess
import os
from tqdm import tqdm
from core.models import LHCbPaper
import time
import shutil
from loguru import logger

import settings

class InspireClient:
    """Client for interacting with the INSPIRE-HEP API."""

    def __init__(
        self,
        abstract_dir: Path = settings.ABSTRACT_DIR,
        pdf_dir: Path = settings.PDF_DIR,
        source_dir: Path = settings.SOURCE_DIR,
        raw_tex_dir: Path = settings.RAW_TEX_DIR,
    ) -> None:
        """Initialize the INSPIRE-HEP client.

        Parameters
        ------------
        abstract_dir : Path
            Directory for storing paper abstracts
        pdf_dir : Path
            Directory for storing PDF versions
        source_dir : Path
            Directory for storing LaTeX source files
        raw_tex_dir : Path
            Directory for storing expanded LaTeX files
        """

        self.base_url = "https://inspirehep.net/api"
        self.abstract_dir = abstract_dir
        self.pdf_dir = pdf_dir
        self.source_dir = source_dir
        self.raw_tex_dir = raw_tex_dir

        for directory in [
            self.abstract_dir,
            self.pdf_dir,
            self.source_dir,
            self.raw_tex_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> "InspireClient":
        """Context manager entry point.

        Returns
        ------------
        InspireClient
            The client instance
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point."""
        pass

    @staticmethod
    def get_arxiv_abstract(abstracts_list: List[Dict[str, Any]]) -> Optional[str]:
        """Extract arXiv abstract if available, otherwise take the first available abstract.

        Parameters
        ----------
        abstracts_list : List[Dict[str, Any]]
            List of abstract dictionaries from INSPIRE API

        Returns
        -------
        Optional[str]
            Abstract text if found, None otherwise
        """
        if not abstracts_list:
            return None

        # Try to find arXiv abstract first
        for abstract in abstracts_list:
            if abstract.get("source") == "arXiv":
                return abstract.get("value")

        # If no arXiv abstract, take the first available one
        return abstracts_list[0].get("value")

    def _fetch_papers(self, params: dict) -> List[LHCbPaper]:
        """Internal method to handle API requests and paper object creation.
        Handles pagination to fetch all results.
        """
        # Copy params so we don't modify the original
        params = params.copy()
        
        # Make initial request to get total number of papers
        response = requests.get(f"{self.base_url}/literature", params=params)
        response.raise_for_status()
        data = response.json()
        
        total_hits = data['hits']['total']
        logger.info(f"Total papers available: {total_hits}")
        
        # Determine how many papers we actually want
        max_results = params.get('size', total_hits)  # If no size specified, get all
        
        # Set page size to API maximum
        params['size'] = settings.REQUEST_CONFIG["max_page_size"]
        
        papers = []
        papers_fetched = 0
        
        # Calculate total pages needed
        total_pages = (min(max_results, total_hits) + 249) // 250  # Ceiling division
        
        for page in tqdm(range(1, total_pages + 1), desc="Fetching LHCb papers from INSPIRE API"):
            params['page'] = page
            response = requests.get(f"{self.base_url}/literature", params=params)
            response.raise_for_status()
            
            hits = response.json()['hits']['hits']
            if not hits:  # No more results
                break
                
            for hit in hits:
                if papers_fetched >= max_results:
                    break
                    
                metadata = hit['metadata']
                arxiv_id = None
                if 'arxiv_eprints' in metadata and metadata['arxiv_eprints']:
                    arxiv_id = metadata['arxiv_eprints'][0].get('value')

                paper = LHCbPaper(
                    title=metadata['titles'][0]['title'],
                    citations=metadata.get('citation_count', 0),
                    arxiv_id=arxiv_id,
                    abstract=self.get_arxiv_abstract(metadata.get('abstracts', [])),
                    arxiv_pdf=f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None,
                    latex_source=f"https://arxiv.org/e-print/{arxiv_id}" if arxiv_id else None
                )
                papers.append(paper)
                papers_fetched += 1
                
            # Break out of page loop if we have enough papers
            if papers_fetched >= max_results:
                break
        
        logger.info(f"Fetching COMPLETE: identified {len(papers)} papers on INSPIRE in TOTAL.")
        return papers

    def fetch_lhcb_papers(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: Optional[int] = None,
        sort_by: str = "mostcited",
    ) -> List[LHCbPaper]:
        """Fetch LHCb collaboration papers from INSPIRE-HEP API.

        Parameters
        ------------
        start_date : str, optional
            Start date in YYYY-MM-DD format
        end_date : str, optional
            End date in YYYY-MM-DD format
        max_results : Optional[int], default=None
            Maximum number of papers to retrieve
        sort_by : str, default='mostcited'
            Sorting method for results ('mostcited', 'mostrecent')

        Returns
        ------------
        List[LHCbPaper]
            List of paper objects matching the search criteria
        """
        query = 'collaboration:"LHCb" and document_type:article'
        if start_date:
            query += f" and date>={start_date}"
        if end_date:
            query += f" and date<={end_date}"

        params = {
            "q": query,
            "sort": sort_by,
            "fields": ["titles,arxiv_eprints,dois,citation_count,abstracts"],
        }

        # if explicit limit is set on number of papers, cap the fetched results
        if max_results is not None:
            params["size"] = max_results

        return self._fetch_papers(params)

    def download_abstract(self, paper: LHCbPaper) -> Optional[Path]:
        """Save paper abstract to file if available.

        Parameters
        ------------
        paper : LHCbPaper
            Paper object containing the abstract text

        Returns
        ------------
        Optional[Path]
            Path to the saved abstract file, or None if abstract not available
        """
        if not paper.abstract:
            return None

        try:
            filepath = self.abstract_dir / f"{paper.arxiv_id}.tex"
            filepath.write_text(paper.abstract)
            return filepath

        except Exception as e:
            logger.error(f"Failed to save abstract: {e}")
            return None

    def download_pdf(self, paper: LHCbPaper) -> Optional[Path]:
        """Download PDF version if available.

        Parameters
        ------------
        paper : LHCbPaper
            Paper object containing the PDF URL

        Returns
        ------------
        Optional[Path]
            Path to the downloaded PDF file, or None if download failed
        """
        if not paper.arxiv_pdf:
            return None

        try:
            response = requests.get(paper.arxiv_pdf)
            response.raise_for_status()

            filepath = self.pdf_dir / f"{paper.arxiv_id}.pdf"
            filepath.write_bytes(response.content)
            return filepath

        except requests.RequestException as e:
            logger.error(f"Failed to download PDF: {e}")
            return None

    def find_main_tex(self, directory: Path) -> Optional[Path]:
        """Find the main TeX file in a directory.

        Parameters
        ------------
        directory : Path
            Directory containing TeX files

        Returns
        ------------
        Optional[Path]
            Path to the main TeX file, or None if not found
        """
        tex_files = list(directory.glob("*.tex"))

        main_candidates = ["main.tex", "paper.tex", "article.tex"]
        for candidate in main_candidates:
            main_file = directory / candidate
            if main_file in tex_files:
                return main_file

        for tex_file in tex_files:
            content = tex_file.read_text(errors="ignore")
            if r"\documentclass" in content:
                return tex_file

        return None

    def extract_and_expand_latex(
        self, paper: LHCbPaper, source_file: Path
    ) -> Optional[Path]:
        """Extract and expand LaTeX source into a single file.

        Parameters
        ------------
        paper : LHCbPaper
            Paper object to process
        source_file : Path
            Path to the downloaded source tarball

        Returns
        ------------
        Optional[Path]
            Path to the expanded LaTeX file, or None if processing failed
        """

        if not source_file.exists():
            logger.error(f"Source file not found: {source_file}")
            return None
            
        # Verify file isn't empty
        if source_file.stat().st_size == 0:
            logger.error(f"Source file is empty: {source_file}")
            return None

        paper_dir = self.source_dir / f"{paper.arxiv_id}"
        paper_dir.mkdir(exist_ok=True)

        try:
            with tarfile.open(source_file) as tar:
                # Check for suspicious paths before extraction
                for member in tar.getmembers():
                    if member.name.startswith('/') or '..' in member.name:
                        logger.error(f"Suspicious path in tarfile: {member.name}")
                        return None
                tar.extractall(path=paper_dir)

            source_file.unlink(missing_ok=True)
        except tarfile.ReadError as e:
            logger.error(f"Failed to extract source: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during extraction: {e}")
            return None

        main_tex = self.find_main_tex(paper_dir)
        if not main_tex:
            logger.error(f"Could not find main TeX file for {paper.arxiv_id}")
            return None

        raw_tex = (self.raw_tex_dir / f"{paper.arxiv_id}.tex").resolve()

        cwd = Path.cwd()
        try:
            os.chdir(main_tex.parent)
            
            # Check if latexpand is available
            if not shutil.which('latexpand'):
                logger.error("latexpand command not found. Please install it.")
                return None
                
            result = subprocess.run(
                ["latexpand", main_tex.name],
                capture_output=True,
                text=True,
                check=True,
                timeout=60  # Add timeout to prevent hanging
            )
            
            if not result.stdout.strip():
                logger.error("latexpand produced empty output")
                return None
                
            raw_tex.write_text(result.stdout)
            return raw_tex

        except subprocess.CalledProcessError as e:
            logger.error(f"latexpand failed: {e.stderr}")
            return None
        except subprocess.TimeoutExpired:
            logger.error("latexpand timed out")
            return None
        except Exception as e:
            logger.error(f"Failed to process LaTeX source: {e}")
            return None
        finally:
            os.chdir(cwd)

    def download_paper_source(self, paper: LHCbPaper) -> Optional[Path]:
        """Download and process paper source files.

        Parameters
        ------------
        paper : LHCbPaper
            Paper object containing the source URL

        Returns
        ------------
        Optional[Path]
            Path to the expanded LaTeX file, or None if processing failed
        """
        return self.download_source(paper)

    def download_source(self, paper: LHCbPaper) -> Optional[Path]:
        """Download paper source files using arXiv's API endpoint.

        Parameters
        ------------
        paper : LHCbPaper
            Paper object containing the source URL

        Returns
        ------------
        Optional[Path]
            Path to the expanded LaTeX file, or None if processing failed
        """
        if not paper.latex_source or not paper.arxiv_id:
            return None

        try:
            # Use arXiv's API endpoint instead of direct download
            source_url = f"http://export.arxiv.org/e-print/{paper.arxiv_id}"
            headers = {
                'User-Agent': 'LHCbPaperBot/1.0 (Contact: your.email@example.com)',
                'Accept': 'application/x-tar, application/x-gzip, */*'
            }

            # Add delay to respect rate limits
            time.sleep(settings.REQUEST_CONFIG["delay_between_requests"])  # Wait between requests

            response = requests.get(
                source_url,
                timeout=settings.REQUEST_CONFIG["timeout"],
                verify=True,
                headers=headers
            )
            response.raise_for_status()

            # Check for CAPTCHA or HTML response
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' in content_type or response.content[:100].decode('utf-8', errors='ignore').strip().startswith(('<html', '<!DOCTYPE')):
                logger.error(f"Received CAPTCHA or HTML instead of tar.gz for {paper.arxiv_id}")
                logger.debug(f"Headers: {response.headers}")
                logger.debug(f"Content start: {response.content[:200]}")
                return None

            # File size checks
            content_length = len(response.content)
            if content_length == 0:
                logger.error("Received empty file")
                return None
            if content_length < 1000:  
                logger.error(f"Suspiciously small file: {content_length} bytes")
                return None
            if content_length > 50 * 1024 * 1024:
                logger.error(f"File too large: {content_length / 1024 / 1024:.1f}MB")
                return None

            source_file = self.source_dir / f"{paper.arxiv_id}_source.tar.gz"
            source_file.write_bytes(response.content)

            # Add exponential backoff for retries
            max_retries = settings.REQUEST_CONFIG["max_retries"]
            for attempt in range(max_retries):
                try:
                    expanded_file = self.extract_and_expand_latex(paper, source_file)
                    if expanded_file:
                        return expanded_file
                except Exception as e:
                    wait_time = 2 ** attempt  # 1, 2, 4 seconds
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        raise

            return None

        except requests.Timeout:
            logger.error("Download timed out")
            return None
        except requests.RequestException as e:
            logger.error(f"Failed to download source: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            return None
