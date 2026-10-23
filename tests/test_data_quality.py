import json
import re
from pathlib import Path
from urllib.parse import unquote

import pytest

from src.data_collection import normalize_text, parse_bcb_date

RAW_PATH = Path("data/raw/raw_communications.json")
CLEAN_PATH = Path("data/processed/communications_clean.json")

MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

def load_json(path: Path) -> list:
    if not path.exists():
        pytest.skip(f"{path} not found")
    return json.loads(path.read_text(encoding="utf-8"))

def test_date_field_matches_date_text():
    """The ISO 'date' must agree with the human-readable 'date_text'."""
    for item in load_json(RAW_PATH):
        parsed = parse_bcb_date(item["date_text"])
        assert parsed is not None, f"Unparseable date_text: {item['date_text']}"
        assert item["date"] == parsed.strftime("%Y-%m-%d"), item["title"]

def test_date_matches_pdf_filename():
    """When the PDF filename contains day + month (e.g. 'NP 15-30 JULIO'), they must match 'date'."""
    pattern = re.compile(r"(\d{1,2})[ _-]*(" + "|".join(MONTHS) + ")")
    for item in load_json(RAW_PATH):
        filename = unquote(item["pdf_url"]).split("/")[-1].lower()
        match = pattern.search(filename)
        if not match:
            continue
        day, month = int(match.group(1)), MONTHS[match.group(2)]
        assert item["date"][5:] == f"{month:02d}-{day:02d}", f"{item['date']} vs {filename}"

def test_clean_file_has_no_duplicates():
    """No two cleaned documents may share the same date and normalized title."""
    keys = [(i["date"], normalize_text(i["title"])) for i in load_json(CLEAN_PATH)]
    assert len(keys) == len(set(keys))

def test_clean_file_has_expected_size():
    """Guard against accidentally dropping or re-adding duplicates."""
    assert len(load_json(CLEAN_PATH)) == 42