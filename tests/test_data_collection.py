import io
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.data_collection import (
    download_official_exchange_rate,
    extract_pdf_text,
    parse_bcb_date,
    scrape_communications_page,
)

# Test Date Parsing 
def test_parse_bcb_date_valid_formats():
    """Test parsing of standard Spanish date formats used by the BCB website."""
    # Pre-clean string with replaced commas to match scraper normalization
    assert parse_bcb_date("15 de marzo de 2023") == datetime(2023, 3, 15)
    assert parse_bcb_date("04 Diciembre 2025") == datetime(2025, 12, 4)
    assert parse_bcb_date("1 de enero 2026") == datetime(2026, 1, 1)

def test_parse_bcb_date_invalid_formats():
    """Test resilience against missing, empty, or unparseable date strings."""
    assert parse_bcb_date("") is None
    assert parse_bcb_date(None) is None
    assert parse_bcb_date("Non-date string content") is None
    assert parse_bcb_date("32 de diciembre de 2025") is None

# Test Dynamic Exchange Rate CSV Download
@patch("src.data_collection.requests.get")
def test_download_official_exchange_rate_success(mock_get, tmp_path, monkeypatch):
    """
    Test downloading and dynamically standardizing CSV columns when BCB headers change.
    """
    monkeypatch.setattr("src.data_collection.DATA_RAW_DIR", tmp_path)

    mock_csv_data = "Fecha_Registro;Tipo_Cambio_Valor\n2026-01-01;6.96\n2026-01-02;6.96"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = mock_csv_data
    mock_get.return_value = mock_response

    output_file = download_official_exchange_rate()

    assert output_file.exists()
    df = pd.read_csv(output_file)
    
    assert list(df.columns) == ["date", "value"]
    assert len(df) == 2


@patch("src.data_collection.requests.get")
def test_download_official_exchange_rate_invalid_headers(mock_get):
    """Test that a ValueError is raised when required date/value columns are missing."""
    mock_csv_data = "UnrelatedCol1;UnrelatedCol2\n1;2"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = mock_csv_data
    mock_get.return_value = mock_response

    with pytest.raises(ValueError, match="Could not identify date and value columns"):
        download_official_exchange_rate()

# Test PDF Text Extraction Safeguard 
@patch("src.data_collection.requests.get")
@patch("src.data_collection.pdfplumber.open")
def test_extract_pdf_text_scanned_warning(mock_pdfplumber, mock_get, capsys):
    """
    Test warning emission when PDF text is suspiciously short (< 50 chars), 
    indicating a scanned image or empty PDF.
    """
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = b"%PDF-mock-bytes"

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Scanned image sample"
    
    mock_pdf = MagicMock()
    mock_pdf.pages = [mock_page]
    mock_pdfplumber.return_value.__enter__.return_value = mock_pdf

    extracted_text = extract_pdf_text("https://www.bcb.gob.bo/sample_scanned.pdf")

    captured = capsys.readouterr()
    assert "Warning: PDF appears to be scanned or empty" in captured.out
    assert extracted_text == "Scanned image sample"

# Test Web Scraping Network Error Resilience 
@patch("src.data_collection.requests.get")
def test_scrape_communications_page_network_error(mock_get):
    """Test that network failures during page scraping return an empty list safely."""
    # Raise requests.RequestException to match the scraper's try...except block
    mock_get.side_effect = requests.RequestException("Connection timeout error")

    items = scrape_communications_page(page_num=0, doc_type="comunicados")
    assert items == []