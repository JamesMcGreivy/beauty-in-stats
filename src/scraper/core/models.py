from typing import List, Optional
from pydantic import BaseModel, Field

class LHCbPaper(BaseModel):
    """Model representing an LHCb paper with its metadata."""
    
    title: str = Field(..., description="Paper title")
    arxiv_id: Optional[str] = Field(None, description="arXiv identifier if available")  # ← FIXED
    citations: int = Field(0, description="Number of citations")
    working_groups: Optional[List[str]] = Field(default_factory=list, description="Associated working groups")
    data_taking_years: Optional[List[str]] = Field(default_factory=list, description="Data-taking years")
    run_period: Optional[str] = Field("<unk>", description="Run period")
    abstract: Optional[str] = Field(None, description="Paper abstract")  # ← FIXED
    latex_source: Optional[str] = Field(None, description="LaTeX source code")
    arxiv_pdf: Optional[str] = Field(None, description="arXiv PDF URL")