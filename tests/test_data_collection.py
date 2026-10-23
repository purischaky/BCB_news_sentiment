from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.data_collection import (
    deduplicate_communications,
    download_official_exchange_rate,
    extract_pdf_text,
    normalize_text,
    parse_bcb_date,
    scrape_communications_page,
)

def make_mock_session(text=None, content=None):
    """Build a fake session whose .get() returns a fake response."""
    response = MagicMock()
    response.status_code = 200
    if text is not None:
        response.text = text
    if content is not None:
        response.content = content
    session = MagicMock()
    session.get.return_value = response
    return session

# Date parsing 
def test_parse_bcb_date_valid_formats():
    assert parse_bcb_date("15 de marzo de 2023") == datetime(2023, 3, 15)
    assert parse_bcb_date("04 Diciembre 2025") == datetime(2025, 12, 4)
    assert parse_bcb_date("1 de enero 2026") == datetime(2026, 1, 1)

def test_parse_bcb_date_invalid_formats():
    assert parse_bcb_date("") is None
    assert parse_bcb_date(None) is None
    assert parse_bcb_date("Non-date string content") is None
    assert parse_bcb_date("32 de diciembre de 2025") is None

# Exchange rate download 
@patch("src.data_collection.create_robust_session")
def test_download_official_exchange_rate_success(mock_create, tmp_path, monkeypatch):
    monkeypatch.setattr("src.data_collection.DATA_RAW_DIR", tmp_path)
    mock_create.return_value = make_mock_session(
        text="Fecha_Registro;Tipo_Cambio_Valor\n2026-01-01;6.96\n2026-01-02;6.96"
    )

    output_file = download_official_exchange_rate()

    assert output_file.exists()
    df = pd.read_csv(output_file)
    assert list(df.columns) == ["date", "value"]
    assert len(df) == 2

@patch("src.data_collection.create_robust_session")
def test_download_official_exchange_rate_invalid_headers(mock_create):
    mock_create.return_value = make_mock_session(text="UnrelatedCol1;UnrelatedCol2\n1;2")

    with pytest.raises(ValueError, match="Could not identify date and value columns"):
        download_official_exchange_rate()

# PDF extraction
@patch("src.data_collection.pdfplumber.open")
def test_extract_pdf_text_scanned_warning(mock_pdfplumber, capsys):
    session = make_mock_session(content=b"%PDF-mock-bytes")

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Scanned image sample"
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdfplumber.return_value.__enter__.return_value = mock_pdf

    text = extract_pdf_text("https://www.bcb.gob.bo/sample_scanned.pdf", session=session)

    assert "Warning: PDF appears to be scanned or empty" in capsys.readouterr().out
    assert text == "Scanned image sample"

# Scraping resilience 
def test_scrape_communications_page_network_error():
    session = MagicMock()
    session.get.side_effect = requests.RequestException("Connection timeout error")

    items = scrape_communications_page(page_num=0, doc_type="comunicados", session=session)

    assert items == []
    
# Deduplication 
def test_normalize_text_removes_accents_prefix_and_case():
    assert normalize_text("COMUNICADO - Revocatoria de Testimonios") == "revocatoria de testimonios"
    assert normalize_text("Difusión de Resultados") == "difusion de resultados"

def test_deduplicate_removes_same_document_with_different_urls():
    items = [
        {"title": "REVOCATORIA DE TESTIMONIOS", "date": "2026-03-05",
         "pdf_url": "https://bcb.gob.bo/a.pdf", "type": "notas-prensa"},
        {"title": "Revocatoria de Testimonios", "date": "2026-03-05",
         "pdf_url": "https://bcb.gob.bo/b.pdf", "type": "comunicados"},
    ]
    assert len(deduplicate_communications(items)) == 1

def test_deduplicate_keeps_same_title_on_different_dates():
    items = [
        {"title": "REVOCATORIA DE TESTIMONIOS", "date": "2026-03-05",
         "pdf_url": "https://bcb.gob.bo/a.pdf", "type": "notas-prensa"},
        {"title": "REVOCATORIA DE TESTIMONIOS", "date": "2026-02-11",
         "pdf_url": "https://bcb.gob.bo/b.pdf", "type": "notas-prensa"},
    ]
    assert len(deduplicate_communications(items)) == 2