import click
from loguru import logger
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Sequence
from dataclasses import dataclass

import settings
from scraper.api_clients.inspire import InspireClient, LHCbPaper
from scraper.builder.post_process_latex import clean_and_expand_macros

@dataclass(frozen=True, slots=True)
class CorpusConfig:
    """Configuration for corpus building process."""

    start_date: Optional[str]
    end_date: Optional[str]
    max_papers: int
    verbose: bool

class CorpusBuilder:
    """Class orchestrating the building an LHCb paper corpus from scraper API."""

    def __init__(self, config: CorpusConfig):
        self.config = config
        self._setup_logger()
        self.client = self._init_inspire_client()

    def _setup_logger(self):
        """Configure logger based on elected verbosity level."""
        log_level = "DEBUG" if self.config.verbose else "INFO"
        logger.remove()
        logger.add(
            settings.LOG_DIR,
            rotation="1 week",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        )
        logger.add(lambda msg: click.echo(msg, err=True), level=log_level)

    def _init_inspire_client(self) -> InspireClient:
        """Initialize INSPIRE client with configured directories."""

        return InspireClient(
            abstract_dir=settings.ABSTRACT_DIR,
            pdf_dir=settings.PDF_DIR,
            source_dir=settings.SOURCE_DIR,
            raw_tex_dir=settings.RAW_TEX_DIR,
        )

    def download_paper(self, paper: LHCbPaper) -> bool:
        """Download PDF and LaTeX source for a single paper.

        Parameters
        ----------
        paper: LHCbPaper
            Paper object to download, defined in scraping.inspire module

        Returns
            bool: True if either PDF or source was successfully downloaded
        """
        success = False

        if paper.arxiv_pdf:
            pdf_path = self.client.download_pdf(paper)
            if pdf_path:
                logger.debug(f"Downloaded PDF: {pdf_path}")
                success = True

        if paper.latex_source:
            source_path = self.client.download_paper_source(paper)
            if source_path:
                logger.debug(f"Downloaded and expanded source: {source_path}")
                success = True

        if paper.abstract:
            abstract_path = self.client.download_abstract(paper)

            if abstract_path:
                logger.debug(f"Downloaded abstract: {abstract_path}")
                success = True

        return success

    def build(self) -> None:
        """Enact the building of the corpus, downloading PDF and latexpanded TeX source."""
        logger.info(
            f"Starting corpus build: max_papers={self.config.max_papers}, "
            f"output_dir={settings.DATA_DIR}"
        )

        papers: Sequence[LHCbPaper] = self.client.fetch_lhcb_papers(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            max_results=self.config.max_papers,
            sort_by="mostcited",
        )

        logger.info(f"Found {len(papers)} papers on INSPIRE")

        if not papers:
            logger.warning("No papers found matching the query criteria, exiting.")
            return

        failed_downloads = []
        for paper in tqdm(papers, desc="Downloading and unpacking LHCb papers"):
            logger.info(
                f"Fetched LHCb paper '{paper.title}' (arXiv:{paper.arxiv_id}) [{paper.citations} citations to date]"
            )

            try:
                success = self.download_paper(paper)
                if not success:
                    failed_downloads.append(paper)
                    logger.warning(
                        f"Failed to download paper '{paper.title}' "
                        f"(arXiv:{paper.arxiv_id})"
                    )
            except Exception as e:
                failed_downloads.append(paper)
                logger.error(
                    f"Error downloading paper '{paper.title}' "
                    f"(arXiv:{paper.arxiv_id}): {str(e)}"
                )

        logger.info(
            f"Successfully downloaded {len(papers) - len(failed_downloads)}/{len(papers)} papers"
        )

        if failed_downloads:
            failed_titles = [
                f"'{p.title}' (arXiv:{p.arxiv_id})" for p in failed_downloads
            ]
            logger.warning(
                f"Failed to download {len(failed_downloads)} papers:\n"
                + "\n".join(f"- {title}" for title in failed_titles)
            )

def validate_date(ctx: click.Context, param: click.Parameter, value: Optional[str]) -> Optional[str]:
    """Validate date format (YYYY-MM-DD)."""
    if not value:
        return None

    try:
        year, month, day = value.split("-")
        assert len(year) == 4 and len(month) == 2 and len(day) == 2
        assert 1900 <= int(year) <= 2100
        assert 1 <= int(month) <= 12
        assert 1 <= int(day) <= 31
        return value
    except (ValueError, AssertionError):
        raise click.BadParameter("Date must be in YYYY-MM-DD format")


@click.command()
@click.option(
    "--start-date", "-s",
    default=settings.CLI_DEFAULTS.get("start_date"),
    help="Start date for paper search (YYYY-MM-DD)",
    callback=validate_date,
)
@click.option(
    "--end-date", "-e", 
    default=settings.CLI_DEFAULTS.get("end_date"),
    help="End date for paper search (YYYY-MM-DD)",
    callback=validate_date,
)
@click.option(
    "--max-papers", "-n",
    default=settings.CLI_DEFAULTS.get("max_papers"),
    type=click.IntRange(1, 10_000),
    help="Maximum number of papers to download",
)
@click.option(
    "--verbose", "-v",
    default=settings.CLI_DEFAULTS.get("verbose"),
    type=bool,
    help="Controls the verbosity of the scraper",
)
def main(**kwargs) -> None:
    """Build an LHCb paper corpus from INSPIRE-HEP.

    Fetches LHCb collaboration papers from INSPIRE-HEP and optionally downloads their
    PDFs and LaTeX sources. Papers can be filtered by date range and are stored in a
    structured directory layout.
    """
    if (
        kwargs.get("start_date") and kwargs.get("end_date") and 
        kwargs["start_date"] > kwargs["end_date"]
    ):
        raise click.BadParameter("Start date must be before end date")

    config = CorpusConfig(**kwargs)
    builder = CorpusBuilder(config)
    builder.build()

    clean_and_expand_macros(settings.RAW_TEX_DIR, settings.CLEAN_TEX_DIR, settings.SECTIONS_TO_REMOVE)


if __name__ == "__main__":
    main()
