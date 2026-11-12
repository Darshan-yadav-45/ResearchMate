"""Academic paper search service using genuine OpenAlex API."""

import logging
import requests
from typing import List, Dict, Any, Optional

from config import Config

logger = logging.getLogger(__name__)


class AcademicPaperSearchService:
    """Queries legitimate academic publications from OpenAlex."""

    OPENALEX_URL = "https://api.openalex.org/works"

    def __init__(self):
        self.headers = {
            "User-Agent": Config.USER_AGENT,
            "Accept": "application/json",
        }

    def search_papers(
        self,
        query: str,
        from_year: Optional[int] = None,
        to_year: Optional[int] = None,
        per_page: int = 15,
        page: int = 1,
    ) -> Dict[str, Any]:
        """Search academic papers using OpenAlex.

        Args:
            query: Natural-language research topic or keywords.
            from_year: Filter papers published from this year onwards.
            to_year: Filter papers published up to this year.
            per_page: Number of papers to return (default 15).
            page: Pagination page number.

        Returns:
            Dict containing papers list, total_count, and pagination info.
        """
        if not query or not query.strip():
            return {"papers": [], "total_count": 0, "page": 1}

        # Construct filters
        filters = []
        if from_year and to_year:
            filters.append(f"publication_year:{from_year}-{to_year}")
        elif from_year:
            filters.append(f"from_publication_date:{from_year}-01-01")
        elif to_year:
            filters.append(f"to_publication_date:{to_year}-12-31")

        # OpenAlex query parameters
        params = {
            "search": query.strip(),
            "per_page": min(per_page, 50),
            "page": page,
        }
        if filters:
            params["filter"] = ",".join(filters)

        try:
            logger.info(f"Querying OpenAlex with params: {params}")
            response = requests.get(
                self.OPENALEX_URL,
                params=params,
                headers=self.headers,
                timeout=20,
            )

            if response.status_code != 200:
                logger.error(f"OpenAlex returned {response.status_code}: {response.text}")
                return {
                    "papers": [],
                    "total_count": 0,
                    "error": f"Academic search API responded with status {response.status_code}",
                }

            data = response.json()
            results = data.get("results", [])
            meta = data.get("meta", {})
            total_count = meta.get("count", 0)

            formatted_papers = []
            for item in results:
                paper = self._format_openalex_item(item)
                if paper:
                    formatted_papers.append(paper)

            return {
                "papers": formatted_papers,
                "total_count": total_count,
                "page": page,
                "per_page": per_page,
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error querying academic search API: {e}")
            return {
                "papers": [],
                "total_count": 0,
                "error": f"Network connection to academic search API failed: {str(e)}",
            }

    def _format_openalex_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parses an OpenAlex work into standardized academic paper schema."""
        if not item:
            return None

        # Authors
        authorships = item.get("authorships", [])
        author_names = []
        for a in authorships:
            author_obj = a.get("author", {})
            name = author_obj.get("display_name")
            if name:
                author_names.append(name)
        authors_str = ", ".join(author_names) if author_names else "Unknown Authors"

        # Abstract inversion (OpenAlex stores abstracts as inverted index)
        abstract = self._reconstruct_abstract(item.get("abstract_inverted_index"))

        # Primary location / Journal
        primary_loc = item.get("primary_location") or {}
        source_obj = primary_loc.get("source") or {}
        journal_name = source_obj.get("display_name", "Academic Venue")

        # DOI & URLs
        doi = item.get("doi") or ""
        landing_url = primary_loc.get("landing_page_url") or doi or item.get("id") or ""
        pdf_url = primary_loc.get("pdf_url") or ""

        # Open access status
        open_access = item.get("open_access", {})
        is_oa = open_access.get("is_oa", False)

        return {
            "title": item.get("title") or "Untitled Academic Work",
            "authors": authors_str,
            "year": item.get("publication_year") or "N/A",
            "abstract": abstract or "No abstract provided in public catalog entry.",
            "doi": doi,
            "url": landing_url,
            "pdf_url": pdf_url,
            "journal": journal_name,
            "citation_count": item.get("cited_by_count", 0),
            "is_open_access": is_oa,
            "source": "openalex",
            "openalex_id": item.get("id"),
        }

    def _reconstruct_abstract(
        self, inverted_index: Optional[Dict[str, List[int]]]
    ) -> str:
        """Reconstructs text from OpenAlex inverted index representation."""
        if not inverted_index:
            return ""

        words_by_pos = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                words_by_pos[pos] = word

        sorted_positions = sorted(words_by_pos.keys())
        return " ".join([words_by_pos[pos] for pos in sorted_positions])
