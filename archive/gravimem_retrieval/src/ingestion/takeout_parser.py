"""
Ingestion & Parser for Google Takeout YouTube Search and Watch History.
"""

import os
import re
import json
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional
from dateutil import parser as date_parser

from src.config import config


class TakeoutParser:
    """Parses Google Takeout search-history.html and watch-history.html."""

    # Patterns to exclude non-informative or shorts noise
    NOISE_PATTERNS = [
        re.compile(r'#shorts', re.IGNORECASE),
        re.compile(r'\bshorts\b', re.IGNORECASE),
        re.compile(r'\btiktok\b', re.IGNORECASE),
        re.compile(r'^\s*$', re.IGNORECASE),
    ]

    def __init__(self, raw_dir: Optional[Path] = None):
        self.raw_dir = raw_dir or config.raw_dir

    def find_takeout_files(self) -> Dict[str, Optional[Path]]:
        """Locates search-history and watch-history files under raw_dir."""
        search_file = None
        watch_file = None

        for root, _, files in os.walk(self.raw_dir):
            for f in files:
                full_path = Path(root) / f
                lower = f.lower()
                if "search-history" in lower and (lower.endswith(".html") or lower.endswith(".json")):
                    search_file = full_path
                elif "watch-history" in lower and (lower.endswith(".html") or lower.endswith(".json")):
                    watch_file = full_path

        return {
            "search_file": search_file,
            "watch_file": watch_file
        }

    def _parse_timestamp(self, date_str: str) -> Optional[datetime]:
        """Converts diverse Takeout date formats into UTC datetime."""
        if not date_str:
            return None
        try:
            # Clean non-breaking spaces, timezone strings
            clean_str = date_str.replace('\u202f', ' ').replace('\xa0', ' ').strip()
            dt = date_parser.parse(clean_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    def _is_noise(self, text: str) -> bool:
        """Determines if a search query is noise / shorts / empty."""
        text_clean = text.strip()
        if len(text_clean) < 2:
            return True
        for pattern in self.NOISE_PATTERNS:
            if pattern.search(text_clean):
                return True
        return False

    def parse_search_history_html(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parses Google Takeout search-history.html into structured events."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Each entry is in a content-cell
        cells = re.findall(
            r'<div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">(.*?)</div>',
            content,
            re.DOTALL
        )

        events = []
        for cell in cells:
            # Match: Searched for <a href="...">Query</a>
            q_match = re.search(r'Searched for\s+<a[^>]*>(.*?)</a>', cell, re.DOTALL)
            text_lines = [re.sub(r'<[^>]+>', '', line).strip() for line in cell.split('<br>') if line.strip()]

            query = None
            date_str = None

            if q_match:
                query = html.unescape(re.sub(r'<[^>]+>', '', q_match.group(1)).strip())
            elif text_lines:
                query = html.unescape(text_lines[0])

            if len(text_lines) > 1:
                date_str = text_lines[-1]

            if not query:
                continue

            query = re.sub(r'^Searched for\s+', '', query).strip()
            if self._is_noise(query):
                continue

            dt = self._parse_timestamp(date_str) if date_str else None

            events.append({
                "type": "search",
                "query": query,
                "timestamp_str": date_str,
                "timestamp_iso": dt.isoformat() if dt else None,
                "epoch": dt.timestamp() if dt else 0.0
            })

        # Sort chronologically (oldest first)
        events.sort(key=lambda x: x["epoch"])
        return events

    def parse_watch_history_html(self, file_path: Path, max_entries: int = 5000) -> List[Dict[str, Any]]:
        """Parses Google Takeout watch-history.html into structured events."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        cells = re.findall(
            r'<div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">(.*?)</div>',
            content,
            re.DOTALL
        )

        events = []
        for cell in cells:
            w_match = re.search(r'Watched\s+<a[^>]*>(.*?)</a>', cell, re.DOTALL)
            text_lines = [re.sub(r'<[^>]+>', '', line).strip() for line in cell.split('<br>') if line.strip()]

            title = None
            channel = None
            date_str = None

            if w_match:
                title = html.unescape(re.sub(r'<[^>]+>', '', w_match.group(1)).strip())
            elif text_lines:
                title = html.unescape(text_lines[0])

            if len(text_lines) >= 3:
                channel = html.unescape(text_lines[1])
                date_str = text_lines[2]
            elif len(text_lines) == 2:
                date_str = text_lines[1]

            if not title:
                continue

            title = re.sub(r'^Watched\s+', '', title).strip()
            if self._is_noise(title):
                continue

            dt = self._parse_timestamp(date_str) if date_str else None

            events.append({
                "type": "watch",
                "title": title,
                "channel": channel,
                "query": title,  # For unified pipeline compatibility
                "timestamp_str": date_str,
                "timestamp_iso": dt.isoformat() if dt else None,
                "epoch": dt.timestamp() if dt else 0.0
            })

        events.sort(key=lambda x: x["epoch"])
        return events[-max_entries:]

    def process_and_save_stream(self, output_path: Optional[Path] = None, include_watch: bool = False) -> Path:
        """Loads Takeout files, processes into chronological stream, and saves to JSON."""
        files = self.find_takeout_files()
        search_file = files["search_file"]
        
        if not search_file:
            raise FileNotFoundError(f"Could not find search-history file in {self.raw_dir}")

        events = self.parse_search_history_html(search_file)
        print(f"Parsed {len(events)} clean search events from {search_file.name}")

        if include_watch and files["watch_file"]:
            watch_events = self.parse_watch_history_html(files["watch_file"])
            print(f"Parsed {len(watch_events)} watch events from {files['watch_file'].name}")
            events.extend(watch_events)
            events.sort(key=lambda x: x["epoch"])

        out = output_path or (config.processed_dir / "chronological_stream.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2, ensure_ascii=False)

        print(f"Successfully saved {len(events)} chronological events to {out}")
        return out


if __name__ == "__main__":
    parser = TakeoutParser()
    out_file = parser.process_and_save_stream()
