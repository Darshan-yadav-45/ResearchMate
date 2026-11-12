"""Academic reference and citation formatting service (APA 7th, IEEE, BibTeX, RIS)."""

import re
from typing import Dict, Any, List


class ReferenceFormatterService:
    """Formats bibliographic data into genuine academic citation standards."""

    def format_all(self, ref_data: Dict[str, Any]) -> Dict[str, str]:
        """Generates APA, IEEE, BibTeX, and RIS formats for a reference.

        Args:
            ref_data: Dict containing title, authors, year, journal, volume, issue, pages, doi, url.

        Returns:
            Dict containing 'apa', 'ieee', 'bibtex', and 'ris'.
        """
        return {
            "apa": self.format_apa(ref_data),
            "ieee": self.format_ieee(ref_data),
            "bibtex": self.format_bibtex(ref_data),
            "ris": self.format_ris(ref_data),
        }

    def format_apa(self, data: Dict[str, Any]) -> str:
        """Formats reference in APA 7th Edition style.
        Rule: Author, A. A., & Author, B. B. (Year). Title. Journal, Vol(Issue), Pages. https://doi.org/...
        """
        authors_raw = data.get("authors") or "Unknown Author"
        authors_apa = self._format_authors_apa(authors_raw)
        year = data.get("year") or "n.d."
        title = (data.get("title") or "Untitled Work").rstrip(".")
        journal = data.get("journal")
        volume = data.get("volume")
        issue = data.get("issue")
        pages = data.get("pages")
        doi = data.get("doi") or ""
        url = data.get("url") or ""

        parts = [f"{authors_apa} ({year}). {title}."]

        if journal:
            journal_part = f" *{journal}*"
            if volume:
                journal_part += f", *{volume}*"
                if issue:
                    journal_part += f"({issue})"
            if pages:
                journal_part += f", {pages}"
            journal_part += "."
            parts.append(journal_part)

        if doi:
            doi_clean = doi if doi.startswith("http") else f"https://doi.org/{doi.lstrip('/')}"
            parts.append(f" {doi_clean}")
        elif url and url.startswith("http"):
            parts.append(f" {url}")

        return "".join(parts).strip()

    def format_ieee(self, data: Dict[str, Any], citation_num: int = 1) -> str:
        """Formats reference in IEEE Citation style.
        Rule: [1] J. K. Author and A. B. Researcher, "Title," Journal, vol. x, no. y, pp. z, Year, doi: ...
        """
        authors_raw = data.get("authors") or "Unknown Author"
        authors_ieee = self._format_authors_ieee(authors_raw)
        title = (data.get("title") or "Untitled Work").rstrip(".")
        year = data.get("year") or "n.d."
        journal = data.get("journal")
        volume = data.get("volume")
        issue = data.get("issue")
        pages = data.get("pages")
        doi = data.get("doi")

        parts = [f"[{citation_num}] {authors_ieee}, \"{title},\""]

        pub_parts = []
        if journal:
            pub_parts.append(f"*{journal}*")
        if volume:
            pub_parts.append(f"vol. {volume}")
        if issue:
            pub_parts.append(f"no. {issue}")
        if pages:
            pub_parts.append(f"pp. {pages}")
        if year != "n.d.":
            pub_parts.append(str(year))

        if pub_parts:
            parts.append(" " + ", ".join(pub_parts) + ".")

        if doi:
            parts.append(f" doi: {doi}.")

        return "".join(parts).strip()

    def format_bibtex(self, data: Dict[str, Any]) -> str:
        """Generates standard BibTeX entry."""
        authors = data.get("authors") or "Unknown"
        title = data.get("title") or "Untitled"
        year = str(data.get("year") or "2024")
        journal = data.get("journal") or "Academic Publication"
        doi = data.get("doi") or ""
        url = data.get("url") or ""

        # Create citation key: e.g. smith2024title
        first_author = re.split(r"[, ]", authors.strip())[0].lower()
        first_author = re.sub(r"[^a-z]", "", first_author) or "ref"
        first_word_title = re.sub(r"[^a-z]", "", title.lower().split()[0]) if title.split() else "work"
        cite_key = f"{first_author}{year}{first_word_title}"

        # Convert authors to BibTeX 'and' separated
        author_list = [a.strip() for a in re.split(r",| and ", authors) if a.strip()]
        bib_authors = " and ".join(author_list) if author_list else authors

        lines = [
            f"@article{{{cite_key},",
            f"  author    = {{{bib_authors}}},",
            f"  title     = {{{{{title}}}}},",
            f"  journal   = {{{journal}}},",
            f"  year      = {{{year}}},",
        ]
        if data.get("volume"):
            lines.append(f"  volume    = {{{data['volume']}}},")
        if data.get("issue"):
            lines.append(f"  number    = {{{data['issue']}}},")
        if data.get("pages"):
            lines.append(f"  pages     = {{{data['pages']}}},")
        if doi:
            lines.append(f"  doi       = {{{doi}}},")
        if url:
            lines.append(f"  url       = {{{url}}},")
        lines.append("}")

        return "\n".join(lines)

    def format_ris(self, data: Dict[str, Any]) -> str:
        """Generates RIS (Research Information Systems) format for EndNote/Zotero."""
        lines = ["TY  - JOUR"]
        title = data.get("title") or "Untitled"
        lines.append(f"TI  - {title}")

        authors_raw = data.get("authors") or "Unknown"
        author_list = [a.strip() for a in re.split(r",| and ", authors_raw) if a.strip()]
        for a in author_list:
            lines.append(f"AU  - {a}")

        if data.get("year"):
            lines.append(f"PY  - {data['year']}")
        if data.get("journal"):
            lines.append(f"JO  - {data['journal']}")
        if data.get("volume"):
            lines.append(f"VL  - {data['volume']}")
        if data.get("issue"):
            lines.append(f"IS  - {data['issue']}")
        if data.get("pages"):
            pages_split = data["pages"].split("-")
            lines.append(f"SP  - {pages_split[0].strip()}")
            if len(pages_split) > 1:
                lines.append(f"EP  - {pages_split[1].strip()}")
        if data.get("doi"):
            lines.append(f"DO  - {data['doi']}")
        if data.get("url"):
            lines.append(f"UR  - {data['url']}")
        lines.append("ER  - ")

        return "\n".join(lines)

    def _format_authors_apa(self, authors_str: str) -> str:
        """Formats author string into APA style."""
        if not authors_str or authors_str == "Unknown Author":
            return "Unknown Author"

        raw_list = [a.strip() for a in re.split(r",| and ", authors_str) if a.strip()]
        if not raw_list:
            return authors_str

        formatted = []
        for name in raw_list:
            parts = name.split()
            if len(parts) >= 2:
                last_name = parts[-1]
                initials = " ".join([f"{p[0]}." for p in parts[:-1] if p])
                formatted.append(f"{last_name}, {initials}")
            else:
                formatted.append(name)

        if len(formatted) == 1:
            return formatted[0]
        elif len(formatted) == 2:
            return f"{formatted[0]}, & {formatted[1]}"
        elif len(formatted) <= 20:
            return ", ".join(formatted[:-1]) + f", & {formatted[-1]}"
        else:
            return ", ".join(formatted[:19]) + ", ... " + formatted[-1]

    def _format_authors_ieee(self, authors_str: str) -> str:
        """Formats author string into IEEE style."""
        if not authors_str or authors_str == "Unknown Author":
            return "Unknown Author"

        raw_list = [a.strip() for a in re.split(r",| and ", authors_str) if a.strip()]
        if not raw_list:
            return authors_str

        formatted = []
        for name in raw_list:
            parts = name.split()
            if len(parts) >= 2:
                initials = " ".join([f"{p[0]}." for p in parts[:-1] if p])
                last_name = parts[-1]
                formatted.append(f"{initials} {last_name}")
            else:
                formatted.append(name)

        if len(formatted) == 1:
            return formatted[0]
        elif len(formatted) == 2:
            return f"{formatted[0]} and {formatted[1]}"
        else:
            return f"{formatted[0]} et al."
