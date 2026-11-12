"""Research report and academic paper draft generation service."""

import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from config import Config

logger = logging.getLogger(__name__)


class ReportGeneratorService:
    """Manages the generation, formatting, and file export of 13-section academic papers."""

    STANDARD_SECTIONS = [
        "Title",
        "Abstract",
        "Keywords",
        "Introduction",
        "Related Work",
        "Research Problem",
        "Research Objectives",
        "Proposed Methodology",
        "Expected Results vs Actual Results",
        "Discussion",
        "Limitations",
        "Conclusion",
        "References",
    ]

    def parse_markdown_to_sections(self, markdown_text: str) -> Dict[str, str]:
        """Parses IBM Granite generated Markdown into structured section dictionary."""
        sections = {sec: "" for sec in self.STANDARD_SECTIONS}

        # Look for headers like # Title, ## 1. Abstract, ## Abstract, etc.
        pattern = r"(?:^|\n)(#{1,3})\s*(?:\d+\.?\s*)?([^\n]+)\n"
        splits = list(re.finditer(pattern, markdown_text))

        if not splits:
            sections["Introduction"] = markdown_text
            return sections

        for i, match in enumerate(splits):
            header_title = match.group(2).strip()
            start_pos = match.end()
            end_pos = splits[i + 1].start() if (i + 1) < len(splits) else len(markdown_text)
            content = markdown_text[start_pos:end_pos].strip()

            # Map detected header to standard section
            matched_key = self._match_section_name(header_title)
            if matched_key:
                sections[matched_key] = content
            elif i == 0 and not sections["Title"]:
                sections["Title"] = header_title

        return sections

    def _match_section_name(self, header: str) -> Optional[str]:
        """Matches a header string to one of the 13 standard sections."""
        h_lower = header.lower()
        if "title" in h_lower:
            return "Title"
        if "abstract" in h_lower:
            return "Abstract"
        if "keyword" in h_lower:
            return "Keywords"
        if "intro" in h_lower:
            return "Introduction"
        if "related" in h_lower or "literature" in h_lower:
            return "Related Work"
        if "problem" in h_lower:
            return "Research Problem"
        if "objective" in h_lower or "aim" in h_lower:
            return "Research Objectives"
        if "method" in h_lower or "architecture" in h_lower or "design" in h_lower:
            return "Proposed Methodology"
        if "result" in h_lower or "finding" in h_lower:
            return "Expected Results vs Actual Results"
        if "discussion" in h_lower:
            return "Discussion"
        if "limitation" in h_lower:
            return "Limitations"
        if "conclusion" in h_lower:
            return "Conclusion"
        if "reference" in h_lower or "bibliography" in h_lower:
            return "References"
        return None

    def export_report(
        self,
        title: str,
        topic: str,
        markdown_content: str,
        export_format: str = "markdown",
    ) -> Dict[str, str]:
        """Exports the research paper draft to disk in specified format.

        Args:
            title: Paper title.
            topic: Research topic.
            markdown_content: Complete draft text in Markdown.
            export_format: 'markdown', 'html', or 'text'.

        Returns:
            Dict with 'file_path', 'filename', and 'format'.
        """
        export_dir = Path(Config.EXPORT_FOLDER)
        export_dir.mkdir(parents=True, exist_ok=True)

        safe_title = re.sub(r"[^\w\-]", "_", title[:40]).strip("_") or "research_paper"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if export_format == "html":
            filename = f"{safe_title}_{timestamp}.html"
            file_path = export_dir / filename
            html_content = self._convert_to_html(title, markdown_content)
            file_path.write_text(html_content, encoding="utf-8")
        elif export_format == "text":
            filename = f"{safe_title}_{timestamp}.txt"
            file_path = export_dir / filename
            plain_text = re.sub(r"[#*_`]", "", markdown_content)
            file_path.write_text(plain_text, encoding="utf-8")
        else:  # markdown default
            filename = f"{safe_title}_{timestamp}.md"
            file_path = export_dir / filename
            file_path.write_text(markdown_content, encoding="utf-8")

        return {
            "filename": filename,
            "file_path": str(file_path),
            "format": export_format,
        }

    def _convert_to_html(self, title: str, markdown_text: str) -> str:
        """Converts Markdown draft to a clean academic HTML document."""
        # Simple HTML styling suitable for academic reading
        body_html = markdown_text.replace("\n\n", "</p><p>").replace("\n", "<br>")
        body_html = re.sub(r"### (.*?)(?:<br>|</p>)", r"<h3>\1</h3><p>", body_html)
        body_html = re.sub(r"## (.*?)(?:<br>|</p>)", r"<h2>\1</h2><p>", body_html)
        body_html = re.sub(r"# (.*?)(?:<br>|</p>)", r"<h1>\1</h1><p>", body_html)
        body_html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", body_html)
        body_html = re.sub(r"\*(.*?)\*", r"<em>\1</em>", body_html)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title} - ResearchMate</title>
  <style>
    body {{
      font-family: 'Times New Roman', Times, serif;
      font-size: 12pt;
      line-height: 1.6;
      color: #111;
      max-width: 800px;
      margin: 40px auto;
      padding: 0 20px;
    }}
    h1 {{ text-align: center; font-size: 18pt; margin-bottom: 24px; }}
    h2 {{ font-size: 14pt; border-bottom: 1px solid #ccc; padding-bottom: 4px; margin-top: 24px; }}
    h3 {{ font-size: 12pt; margin-top: 18px; }}
    p {{ text-align: justify; margin-bottom: 12px; }}
    .notice {{
      background: #fff3cd;
      border-left: 4px solid #ffc107;
      padding: 8px 12px;
      font-size: 10pt;
      font-family: sans-serif;
      margin: 10px 0;
    }}
  </style>
</head>
<body>
  <p><small>Generated by ResearchMate – Academic Research Platform</small></p>
  <p>{body_html}</p>
</body>
</html>"""
