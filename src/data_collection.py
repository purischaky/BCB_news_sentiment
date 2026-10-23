import io
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import pdfplumber
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

# Configuration & Constants 
DATA_RAW_DIR = Path("data/raw")
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.bcb.gob.bo"
BCB_EVOLUTIVO_BASE = "https://www.bcb.gob.bo/grafico_evolutivo/download.php"

SPANISH_MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

def create_robust_session() -> requests.Session:
    """Creates a requests session equipped with automatic retry and backoff logic."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    return session

# Data Collection Functions
def download_official_exchange_rate() -> Path:
    """
    Downloads the FULL historical official/reference exchange rate series from the BCB
    using a resilient session with automatic retries.
    """
    print("  Fetching FULL historical exchange rate data...")
    session = create_robust_session()
    
    response = session.get(
        BCB_EVOLUTIVO_BASE, 
        params={"serie": "tipo_de_cambio", "format": "csv"}, 
        timeout=15
    )
    response.raise_for_status()
    
    df = pd.read_csv(io.StringIO(response.text), sep=";", encoding="utf-8-sig")
    
    date_cols = [col for col in df.columns if "fecha" in col.lower() or "date" in col.lower()]
    value_cols = [col for col in df.columns if "valor" in col.lower() or "tipo" in col.lower() or "cambio" in col.lower()]
    
    if not date_cols or not value_cols:
        raise ValueError("Could not identify date and value columns in the BCB CSV. Headers might have changed.")
    
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
    Parses Spanish date strings (e.g., '15 de marzo, 2023') into a Python datetime object.
    """
    if not date_text:
        return None
    
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
            return None
            
    return None

def extract_pdf_text(pdf_url: str, session: requests.Session) -> str:
    """
    Downloads a PDF from a URL and extracts its text content using a resilient session.
    """
    if "chrome-extension://" in pdf_url:
        match = re.search(r'(https?://.+)', pdf_url)
        if match:
            pdf_url = match.group(1)  
        else:
            print(f"    ⚠ Invalid URL (extension prefix): {pdf_url}")
            return ""
    
    try:
        response = session.get(pdf_url, timeout=15)
        response.raise_for_status()
        
        text_pages = []
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_pages.append(text)
        
        final_text = "\n".join(text_pages).strip()
        
        if len(final_text) < 50:
            print(f"    ⚠ Warning: PDF appears to be scanned or empty: {pdf_url}")
            
        return final_text
        
    except requests.RequestException as e:
        print(f"    ⚠ Network error fetching PDF ({pdf_url}): {e}")
        return ""
    except Exception as e:
        print(f"    ⚠ Error extracting text from PDF ({pdf_url}): {e}")
        return ""

def scrape_communications_page(page_num: int, doc_type: str, session: requests.Session) -> list[dict]:
    """
    Scrapes a single page of BCB communications or press notes.
    """
    url = f"{BASE_URL}/?q={doc_type}"
    if page_num > 0:
        url += f"&page={page_num}"

    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"    ⚠ Failed to fetch page {page_num} of {doc_type}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    items = []
    
    entries = soup.select("div.view-content .views-row") or soup.select("div.view-content > div")

    for entry in entries:
        link_tag = entry.find("a", href=True)
        if not link_tag:
            continue
            
        title = link_tag.get_text(strip=True)
        pdf_url = urljoin(BASE_URL, link_tag["href"])
        
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
    Iterates through communications with incremental disk-writing (JSON lines) 
    to prevent memory loss if interrupted.
    """
    session = create_robust_session()
    doc_types = ["comunicados", "notas-prensa"]
    temp_jsonl_path = DATA_RAW_DIR / "raw_communications_temp.jsonl"
    
    results = []
    
    # Open temporary JSON lines file for incremental writes
    with open(temp_jsonl_path, "w", encoding="utf-8") as f_out:
        for doc_type in doc_types:
            print(f"  Scraping category '{doc_type}' (stopping at {cutoff_date.strftime('%Y-%m-%d')})...")
            page = 0
            stop_scraping_category = False
            
            while not stop_scraping_category:
                items = scrape_communications_page(page, doc_type=doc_type, session=session)
                if not items:
                    break
                
                valid_parsed_count = 0
                older_items_count = 0
                
                for item in items:
                    parsed_date = parse_bcb_date(item["date_text"])
                    if not parsed_date:
                        continue
                        
                    valid_parsed_count += 1
                    
                    if parsed_date < cutoff_date:
                        older_items_count += 1
                        continue
                    
                    print(f"    Processing: {item['title'][:50]}... ({parsed_date.strftime('%Y-%m-%d')})")
                    
                    time.sleep(0.3)
                    full_text = extract_pdf_text(item["pdf_url"], session=session)
                    
                    record = {
                        "title": item["title"],
                        "pdf_url": item["pdf_url"],
                        "date_text": item["date_text"],
                        "date": parsed_date.strftime("%Y-%m-%d"),
                        "type": doc_type,
                        "full_text": full_text
                    }
                    
                    results.append(record)
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                
                if valid_parsed_count > 0 and older_items_count == valid_parsed_count:
                    stop_scraping_category = True
                    
                page += 1
                
    # Clean up temp file if needed, or keep it. We load from results list.
    if temp_jsonl_path.exists():
        temp_jsonl_path.unlink()
        
    return results

# Deduplication Functions
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def deduplicate_communications(raw_items: list) -> list:
    """
    Deduplicates raw scraped communications by canonical URL or (date + normalized title).
    """
    seen_keys = set()
    unique_items = []
    
    sorted_items = sorted(raw_items, key=lambda x: x.get('type', ''), reverse=True)
    
    for item in sorted_items:
        pdf_url = item.get('pdf_url', '').strip()
        date = item.get('date', '').strip()
        title_norm = normalize_text(item.get('title', ''))
        
        dedup_key = pdf_url if pdf_url else f"{date}::{title_norm}"
        
        if dedup_key not in seen_keys:
            seen_keys.add(dedup_key)
            unique_items.append(item)
            
    return unique_items

# Main Pipeline Execution
def run_collection(start_date: datetime):
    print("=" * 60)
    print("Starting BCB Data Collection Pipeline")
    print("=" * 60)
    
    print("\n1. Downloading FULL historical exchange rate series...")
    download_official_exchange_rate()
    
    print("\n2. Scraping BCB communications incrementally...")
    raw_communications = collect_relevant_communications(cutoff_date=start_date)
    
    print("\n3. Deduplicating communications...")
    communications = deduplicate_communications(raw_communications)
    duplicates_removed = len(raw_communications) - len(communications)
    print(f"  ✓ Removed {duplicates_removed} duplicate(s). {len(communications)} unique documents remain.")
    
    print("\n4. Saving final communications data...")
    output_path = DATA_RAW_DIR / "raw_communications.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(communications, f, ensure_ascii=False, indent=2)
        
    print(f"\n✓ Pipeline Complete: {len(communications)} unique documents saved to {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    one_year_ago = datetime.now() - timedelta(days=365)
    print(f"🎯 Target: Scraping communications from {one_year_ago.strftime('%Y-%m-%d')} to present.")
    run_collection(start_date=one_year_ago)