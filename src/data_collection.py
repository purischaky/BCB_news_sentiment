import io
import json
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup

# Configuration & Constants 
DATA_RAW_DIR = Path("data/raw")
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.bcb.gob.bo"
BCB_EVOLUTIVO_BASE = "https://www.bcb.gob.bo/grafico_evolutivo/download.php"

# Dictionary to map Spanish month names to integers for datetime parsing
SPANISH_MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

def download_official_exchange_rate() -> Path:
    """
    Downloads the historical official/reference exchange rate series from the BCB.
    Uses dynamic column mapping to prevent crashes if the BCB changes CSV headers.
    """
    print("  Fetching official exchange rate data...")
    response = requests.get(
        BCB_EVOLUTIVO_BASE, 
        params={"serie": "tipo_de_cambio", "format": "csv"}, 
        timeout=15
    )
    response.raise_for_status()
    
    df = pd.read_csv(io.StringIO(response.text), sep=";", encoding="utf-8-sig")
    
    # Dynamically identify date and value columns
    date_cols = [col for col in df.columns if "fecha" in col.lower() or "date" in col.lower()]
    value_cols = [col for col in df.columns if "valor" in col.lower() or "tipo" in col.lower() or "cambio" in col.lower()]
    
    if not date_cols or not value_cols:
        raise ValueError("Could not identify date and value columns in the BCB CSV. Headers might have changed.")
    
    # Standardize column names
    df = df[[date_cols[0], value_cols[0]]].rename(
        columns={date_cols[0]: "date", value_cols[0]: "value"}
    )
    df["date"] = pd.to_datetime(df["date"])
    
    file_path = DATA_RAW_DIR / "official_exchange_rate.csv"
    df.to_csv(file_path, index=False)
    print(f"  ✓ Exchange rate saved: {len(df)} records up to {df['date'].max().date()}")
    return file_path

def parse_bcb_date(date_text: str) -> datetime | None:
    """
    Parses Spanish date strings (e.g., '15 de marzo, 2023' or '15 marzo 2023') 
    into a Python datetime object.
    """
    if not date_text:
        return None
    
    # Normalize whitespace and clean commas
    date_text = " ".join(str(date_text).replace(",", "").split())
    
    match = re.search(r"(\d{1,2})\s+(?:de\s+)?(\w+)\s+(?:de\s+)?(\d{4})", date_text, re.IGNORECASE)
    if not match:
        return None

    day, month_text, year = match.groups()
    month = SPANISH_MONTHS.get(month_text.lower().strip())
    
    if month:
        try:
            return datetime(int(year), month, int(day))
        except ValueError:
            # Handles invalid dates like Feb 30 or Dec 32
            return None
            
    return None

def extract_pdf_text(pdf_url: str) -> str:
    """
    Downloads a PDF from a URL and extracts its text content.
    Includes a safeguard to warn if the PDF is likely a scanned image (very little text).
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(pdf_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        text_pages = []
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_pages.append(text)
        
        final_text = "\n".join(text_pages).strip()
        
        # Safeguard: Warn if extracted text is suspiciously short (likely a scanned image)
        if len(final_text) < 50:
            print(f"    ⚠ Warning: PDF appears to be scanned or empty (minimal text extracted): {pdf_url}")
            
        return final_text
        
    except requests.RequestException as e:
        print(f"    ⚠ Network error fetching PDF ({pdf_url}): {e}")
        return ""
    except Exception as e:
        print(f"    ⚠ Error extracting text from PDF ({pdf_url}): {e}")
        return ""

def scrape_communications_page(page_num: int, doc_type: str = "comunicados") -> list[dict]:
    """
    Scrapes a single page of BCB communications or press notes.
    """
    url = f"{BASE_URL}/?q={doc_type}"
    if page_num > 0:
        url += f"&page={page_num}"

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"    ⚠ Failed to fetch page {page_num} of {doc_type}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    items = []
    
    # Target the specific Drupal view rows
    entries = soup.select("div.view-content .views-row") or soup.select("div.view-content > div")

    for entry in entries:
        link_tag = entry.find("a", href=True)
        if not link_tag:
            continue
            
        title = link_tag.get_text(strip=True)
        pdf_url = urljoin(BASE_URL, link_tag["href"])
        
        # Two-tier date extraction strategy
        date_text = ""
        date_span = entry.find("span", class_=re.compile(r"date|fecha|time", re.IGNORECASE))
        if date_span:
            date_text = date_span.get_text(strip=True)
        else:
            full_text = entry.get_text(" ", strip=True)
            match = re.search(r"(\d{1,2})\s+(?:de\s+)?(\w+)\s+(?:de\s+)?(\d{4})", full_text, re.IGNORECASE)
            if match:
                date_text = f"{match.group(1)} {match.group(2)} {match.group(3)}"

        items.append({
            "title": title, 
            "pdf_url": pdf_url, 
            "date_text": date_text
        })
        
    return items

def collect_relevant_communications(cutoff_date: datetime) -> list[dict]:
    """
    Iterates through BCB communications and press notes, skipping entries older 
    than cutoff_date and breaking when a full page of older items is encountered.
    """
    results = []
    doc_types = ["comunicados", "notas-prensa"]
    
    for doc_type in doc_types:
        print(f"  Scraping category '{doc_type}'...")
        page = 0
        stop_scraping_category = False
        
        while not stop_scraping_category:
            items = scrape_communications_page(page, doc_type=doc_type)
            if not items:
                break
            
            older_items_count = 0
            
            for item in items:
                parsed_date = parse_bcb_date(item["date_text"])
                if not parsed_date:
                    continue  # Skip items with unparseable dates
                    
                if parsed_date < cutoff_date:
                    older_items_count += 1
                    continue
                
                print(f"    Processing: {item['title'][:50]}... ({parsed_date.strftime('%Y-%m-%d')})")
                
                # Polite request delay to prevent server rate-limiting
                time.sleep(0.3)
                full_text = extract_pdf_text(item["pdf_url"])
                
                results.append({
                    "title": item["title"],
                    "pdf_url": item["pdf_url"],
                    "date_text": item["date_text"],
                    "date": parsed_date.strftime("%Y-%m-%d"),
                    "type": doc_type,
                    "full_text": full_text
                })
            
            # If all parsed items on the page are older than cutoff_date, stop category pagination
            if items and older_items_count == len(items):
                stop_scraping_category = True
                
            page += 1
            
    return results

def run_collection(start_date: datetime = datetime(2023, 6, 29)):
    """
    Main execution function to orchestrate the data collection pipeline.
    """
    print("=" * 50)
    print("Starting BCB Data Collection Pipeline")
    print("=" * 50)
    
    print("\n1. Downloading official exchange rate series...")
    download_official_exchange_rate()
    
    print("\n2. Scraping BCB communications and press notes...")
    communications = collect_relevant_communications(cutoff_date=start_date)
    
    print("\n3. Saving raw communications data...")
    output_path = DATA_RAW_DIR / "raw_communications.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(communications, f, ensure_ascii=False, indent=2)
        
    print(f"\n✓ Pipeline Complete: {len(communications)} documents saved to {output_path}")
    print("=" * 50)

if __name__ == "__main__":
    run_collection(start_date=datetime(2023, 6, 29))