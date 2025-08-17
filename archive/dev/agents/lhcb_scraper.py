from pydantic import BaseModel
from typing import List, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
import json
import pandas as pd
from datetime import datetime
import re

class LHCbPaper(BaseModel):
    lhcb_paper_id: str
    title: str  
    arxiv_id: str
    journal: str
    working_groups: List[str]
    data_taking_years: List[str]
    run_period: str

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
    """Parse years from cell content, handling empty or 'performance' cases"""
    if not years_cells.strip():
        return []
    return [year.strip() for year in years_cells.split() if year.strip().isdigit()]

def process_page(driver) -> List[LHCbPaper]:
    """Process a single page of papers"""
    page_papers = []
    rows = driver.find_elements(By.TAG_NAME, "tr")
    
    for row in rows:
        cells = row.find_elements(By.TAG_NAME, "td")
        if len(cells) >= 8:  # Make sure we have enough cells
            try:
                # Get the title from the Title column (index 2)
                title = cells[2].text.strip()
                years = parse_years(cells[7].text)
                working_groups = parse_working_groups(cells[6].text)
                
                paper = LHCbPaper(
                    lhcb_paper_id=cells[3].text.strip(),
                    title=title,  # Added title field
                    arxiv_id=cells[4].text.strip(),
                    journal=cells[5].text.strip(),
                    working_groups=working_groups,
                    data_taking_years=years,
                    run_period=determine_run_period(years)
                )
                page_papers.append(paper)
                print(f"Parsed paper: {paper.lhcb_paper_id} - {paper.title[:50]}...")  # Added title to logging
            except Exception as e:
                print(f"Error parsing row: {e}")
                continue
    
    return page_papers

def scrape_papers():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    
    all_papers = []
    
    try:
        print("Loading page...")
        driver.get('https://lbfence.cern.ch/alcm/public/analysis')
        
        wait = WebDriverWait(driver, 10)
        table = wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        
        while True:
            print("\nProcessing current page...")
            page_papers = process_page(driver)
            all_papers.extend(page_papers)
            
            try:
                next_button = driver.find_element(By.XPATH, "//button[@aria-label='Next page']")
                if not next_button or 'disabled' in next_button.get_attribute('class'):
                    print("Reached last page")
                    break
                    
                next_button.click()
                print("Moving to next page...")
                sleep(1)
                
            except Exception as e:
                print("No more pages found")
                break
    
    except Exception as e:
        print(f"Error scraping: {e}")
    
    finally:
        driver.quit()
    
    return all_papers

def save_papers(papers: list):
    """Save papers to pickle format, preserving all data structures"""
    paper_dicts = [paper.dict() for paper in papers]
    pd_df = pd.DataFrame(paper_dicts)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pickle_path = f'lhcb_papers_{timestamp}.pkl'
    
    pd_df.to_pickle(pickle_path)
    print(f"\nSaved to: {pickle_path}")
    
    return pd_df

def main():
    papers = scrape_papers()
    df = save_papers(papers)
    
    print(f"\nSuccessfully scraped {len(papers)} papers")
    print(f"\nDataFrame shape: {df.shape}")
    
    print("\nWorking Group counts:")
    wg_counts = (df.explode('working_groups')
                  .groupby('working_groups')
                  .size()
                  .sort_values(ascending=False)
                  .reset_index(name='count'))
    print(wg_counts)
    
    print("\nRun Period distribution:")
    run_period_counts = (df.groupby('run_period')
                          .size()
                          .sort_values(ascending=False)
                          .reset_index(name='count'))
    print(run_period_counts)
    
    print("\nSample papers:")
    for paper in papers[:3]:
        print("\n---")
        print(f"Title: {paper.title}")  # Added title to sample output
        print(f"LHCb ID: {paper.lhcb_paper_id}")
        print(f"arXiv ID: {paper.arxiv_id}")
        print(f"Journal: {paper.journal}")
        print(f"Working Groups: {', '.join(paper.working_groups)}")
        print(f"Years: {', '.join(paper.data_taking_years)}")
        print(f"Run Period: {paper.run_period}")

if __name__ == "__main__":
    main()