from pathlib import Path
from typing import List, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
from time import sleep
import pandas as pd
import re
from loguru import logger
from api_clients.inspire import InspireClient
from core.models import LHCbPaper
from scripts.build_lhcb_corpus import CorpusBuilder, CorpusConfig
from scripts.post_process_latex import clean_and_expand_macros
from tqdm import tqdm
import requests
import click


def normalize_working_group(wg: str) -> str:
    """Convert working group label to snake_case format"""
    wg = wg.strip().lower()
    wg = re.sub(r'[^a-z0-9]+', '_', wg)
    wg = wg.strip('_')
    return wg

def parse_working_groups(wg_cell: str) -> List[str]:
    """Parse and normalize multiple working groups from a cell"""
    working_groups = [wg.strip() for wg in wg_cell.split('\n') if wg.strip()]
    return [normalize_working_group(wg) for wg in working_groups]

def determine_run_period(years: List[str]) -> str:
    """Map years to LHC run periods"""
    if not years:
        return "unknown"
    
    run_periods = []
    for year in years:
        year = int(year)
        if year in {2011, 2012}:
            run_periods.append("Run1")
        elif year in {2015, 2016, 2017, 2018}:
            run_periods.append("Run2")
        elif year in {2023, 2024, 2025}:
            run_periods.append("Run3")
        else:
            run_periods.append("unknown")
    
    unique_periods = sorted(set(run_periods))
    return "+".join(unique_periods) if unique_periods else "unknown"

def parse_years(years_cells: str) -> List[str]:
    """Parse years from cell content"""
    if not years_cells.strip():
        return []
    return [year.strip() for year in years_cells.split() if year.strip().isdigit()]

def process_page(driver) -> List[dict]:
    """Process a single page of papers"""
    page_papers = []
    rows = driver.find_elements(By.TAG_NAME, "tr")
    
    for row in rows:
        cells = row.find_elements(By.TAG_NAME, "td")
        if len(cells) >= 8:
            try:
                title = cells[2].text.strip()
                arxiv_id = cells[4].text.strip()
                years = parse_years(cells[7].text)
                working_groups = parse_working_groups(cells[6].text)
                
                paper_info = {
                    'title': title,
                    'arxiv_id': arxiv_id,
                    'lhcb_paper_id': cells[3].text.strip(),
                    'journal': cells[5].text.strip(),
                    'working_groups': working_groups,
                    'data_taking_years': years,
                    'run_period': determine_run_period(years)
                }
                page_papers.append(paper_info)
                logger.debug(f"Parsed paper: {paper_info['lhcb_paper_id']} - {title[:50]}...")
            except Exception as e:
                logger.error(f"Error parsing row: {e}")
                continue
    
    return page_papers

def scrape_and_enrich_papers(
    max_papers: Optional[int] = None,
    download: bool = False,
    output_dir: Optional[Path] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Scrape papers from LHCb publication page and enrich with INSPIRE metadata.
    
    Parameters
    ----------
    max_papers : Optional[int]
        Maximum number of papers to process
    download : bool
        Whether to download PDFs and LaTeX sources
    output_dir : Optional[Path]
        Directory to save downloaded files
    verbose : bool
        Enable verbose logging
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing all paper metadata
    """
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--page-load-timeout=30')
    
    driver = None
    papers_data = []
    inspire_client = InspireClient()
    
    try:
        # Add retry logic for driver initialization
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing Chrome WebDriver (attempt {attempt + 1}/{max_retries})")
                driver = webdriver.Chrome(options=options)
                driver.set_page_load_timeout(30)
                break
            except Exception as e:
                logger.error(f"WebDriver initialization failed (attempt {attempt + 1}): {e}")
                if driver:
                    try:
                        driver.quit()
                    except:
                        pass
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)
        
        logger.info("Loading LHCb publication page...")
        # Add retry logic for page load
        max_page_retries = 3
        for attempt in range(max_page_retries):
            try:
                driver.get('https://lbfence.cern.ch/alcm/public/analysis')
                wait = WebDriverWait(driver, 10)
                table = wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
                break
            except Exception as e:
                logger.error(f"Page load failed (attempt {attempt + 1}): {e}")
                if attempt == max_page_retries - 1:
                    raise
                time.sleep(2)
        
        while True:
            page_papers = process_page(driver)
            logger.info(f"Found {len(page_papers)} papers on current page")
            
            for paper in page_papers:
                if not paper['arxiv_id']:
                    continue
                    
                try:
                    # Add timeout to API request
                    query = f'arxiv:{paper["arxiv_id"]}'
                    params = {
                        "q": query,
                        "fields": ["titles,arxiv_eprints,dois,citation_count,abstracts"],
                    }
                    response = requests.get(
                        f"{inspire_client.base_url}/literature",
                        params=params,
                        timeout=10  # Add timeout
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    if data['hits']['hits']:
                        metadata = data['hits']['hits'][0]['metadata']
                        citations = metadata.get('citation_count', 0)
                        abstract = inspire_client.get_arxiv_abstract(metadata.get('abstracts', []))
                        arxiv_eprints = metadata.get('arxiv_eprints', [])
                        arxiv_pdf = None
                        if arxiv_eprints:
                            arxiv_id = arxiv_eprints[0].get('value')
                            if arxiv_id:
                                arxiv_pdf = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                    else:
                        citations = 0
                        abstract = ""
                        arxiv_pdf = None
                        
                except requests.Timeout:
                    logger.warning(f"Timeout while fetching metadata for {paper['arxiv_id']}")
                    citations = 0
                    abstract = ""
                    arxiv_pdf = None
                except Exception as e:
                    logger.warning(f"Failed to fetch metadata for {paper['arxiv_id']}: {e}")
                    citations = 0
                    abstract = ""
                    arxiv_pdf = None
                
                # Create enriched paper object
                arxiv_id = paper['arxiv_id']
                paper_obj = LHCbPaper(
                    lhcb_paper_id=paper['lhcb_paper_id'],
                    title=paper['title'],
                    arxiv_id=arxiv_id,
                    citations=citations,
                    working_groups=paper['working_groups'],
                    data_taking_years=paper['data_taking_years'],
                    run_period=paper['run_period'],
                    abstract=abstract,
                    arxiv_pdf=f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None,
                    latex_source=f"https://arxiv.org/e-print/{arxiv_id}" if arxiv_id else None
                )
                
                papers_data.append(paper_obj.__dict__)
                
                if max_papers and len(papers_data) >= max_papers:
                    logger.info(f"Reached limit of {max_papers} papers")
                    break
            
            if max_papers and len(papers_data) >= max_papers:
                break
            
            # Check for next page with explicit timeout
            try:
                next_button = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, "//button[@aria-label='Next page']"))
                )
                if not next_button or 'disabled' in next_button.get_attribute('class'):
                    logger.info("Reached last page")
                    break
                
                next_button.click()
                time.sleep(1)
                
            except TimeoutException:
                logger.info("No more pages found (timeout)")
                break
            except Exception as e:
                logger.info(f"No more pages found: {e}")
                break
    
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass
    
    # Create DataFrame
    df = pd.DataFrame(papers_data)
    
    # Save data if output_dir provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_pickle(output_dir / 'lhcb_papers.pkl')
        df.to_csv(output_dir / 'lhcb_papers.csv', index=False)
        logger.info(f"Saved paper metadata to {output_dir}")
        
        if download:
            config = CorpusConfig(
                start_date=None,
                end_date=None,
                max_papers=len(df),
                download=True,
                output_dir=output_dir,
                verbose=verbose
            )
            builder = CorpusBuilder(config)
            
            failed_downloads = []
            for _, row in tqdm(df.iterrows(), desc="Downloading and processing papers", total=len(df)):
                paper = LHCbPaper(**row.to_dict())
                
                try:
                    # Download paper
                    success = builder.download_paper(paper)
                    if not success:
                        failed_downloads.append(paper.lhcb_paper_id)
                        continue
                
                except Exception as e:
                    failed_downloads.append(paper.lhcb_paper_id)
                    logger.error(f"Error processing paper '{paper.title}': {str(e)}")
            
            if failed_downloads:
                logger.warning(f"Failed to process {len(failed_downloads)} papers")
    
    # Create cleaned_tex directory
    tex_dir = output_dir / "expanded_tex"
    cleaned_tex_dir = output_dir / "cleaned_tex"
    cleaned_tex_dir.mkdir(parents=True, exist_ok=True)
    sections_to_remove = [
        'Acknowledgements',
        'Acknowledgments',
        'References',
        'Bibliography',
    ]
    clean_and_expand_macros(tex_dir, cleaned_tex_dir, sections_to_remove)
    
    return df

def validate_paper_count(
    ctx: click.Context, param: click.Parameter, value: Optional[int]
) -> Optional[int]:
    """
    Validate paper count is a positive integer.

    Parameters
    ----------
    ctx: click.Context
        Click context
    param: click.Parameter
        Click parameter
    value: Optional[int]
        Number of papers to download

    Returns
    -------
    Optional[int]
        Number of papers to download if positive
    """
    if value is not None and value <= 0:
        raise click.BadParameter("Number of papers must be positive")
    return value

@click.command()
@click.option(
    "--max-papers",
    "-n",
    default=None,
    type=click.IntRange(1, 10_000),
    callback=validate_paper_count,
    help="Maximum number of LHCb papers to scrape (default: no limit)",
)
@click.option(
    "--download/--no-download",
    default=True,
    help="Download PDFs and LaTeX source for papers",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("data"),
    help="Base directory for downloaded files",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(**kwargs) -> None:
    """Scrape LHCb papers and optionally download their content."""
    # Set up logging
    log_level = "DEBUG" if kwargs.get('verbose') else "INFO"
    logger.remove()
    logger.add(
        "scraping.log",
        rotation="1 week",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    )
    logger.add(lambda msg: click.echo(msg, err=True), level=log_level)
    
    # Scrape and process papers
    df = scrape_and_enrich_papers(
        max_papers=kwargs.get('max_papers'),
        download=kwargs.get('download'),
        output_dir=kwargs.get('output_dir'),
        verbose=kwargs.get('verbose')
    )
    
    logger.info(f"Successfully processed {len(df)} papers")

if __name__ == "__main__":
    main()