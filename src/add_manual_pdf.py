import io
import json
from pathlib import Path
import pdfplumber
import requests

DATA_RAW_DIR = Path("data/raw")
JSON_PATH = DATA_RAW_DIR / "raw_communications.json"

# You can define manual entries here to batch-process them safely.
# The script will automatically skip any URL that already exists in raw_communications.json.
MANUAL_ENTRIES = [
    {
        "title": "EL BCB CUMPLE CON LA TOTALIDAD DE SUS OBLIGACIONES EN ORO Y CIERRA LAS TRES OPERACIONES FORWARD CONTRATADAS EN 2025",
        "pdf_url": "https://www.bcb.gob.bo/webdocs/files_noticias/NP%2025-14%20OCTUBRE.pdf",
        "date": "2026-10-14",
        "date_text": "14 de octubre de 2026",
        "type": "notas-prensa"
    },
    {
        "title": "EL BCB ESTABLECE LA RESERVA MONETARIA AMPLIADA PARA ESTERILIZAR PARTE DE LA LIQUIDEZ INYECTADA EN SEPTIEMBRE",
        "pdf_url": "https://www.bcb.gob.bo/webdocs/files_noticias/NP%2023-25%20SEPTIEMBRE.pdf",
        "date": "2026-09-25",
        "date_text": "25 de septiembre de 2026",
        "type": "notas-prensa"
    },
    {
        "title": "BCB OFERTARÁ DÓLARES ESTADOUNIDENSES AL SISTEMA FINANCIERO",
        "pdf_url": "https://www.bcb.gob.bo/webdocs/files_noticias/NP%2022-8%20SEPTIEMBRE%284%29.pdf",
        "date": "2026-09-08",
        "date_text": "08 de septiembre de 2026",
        "type": "notas-prensa"
    },
    {
        "title": "MEDIDAS DE POLÍTICA CAMBIARIA Y MONETARIA",
        "pdf_url": "https://www.bcb.gob.bo/webdocs/files_noticias/NP%2021-7%20SEPTIEMBRE.pdf",
        "date": "2026-09-07",
        "date_text": "07 de septiembre de 2026",
        "type": "notas-prensa"
    },
    {
        "title": "BCB presenta el “Informe de operaciones con reservas en oro-Segundo cuatrimestre de 2026",
        "pdf_url": "https://www.bcb.gob.bo/webdocs/files_noticias/NP%2020-3%20SEPTIEMBRE.pdf",
        "date": "2026-09-03",
        "date_text": "03 de septiembre de 2026",
        "type": "notas-prensa"
    },
    {
        "title": "POLÍTICAS DEL BCB ESTÁN ORIENTADAS A UN MERCADO CAMBIARIO PLENAMENTE FLEXIBLE",
        "pdf_url": "https://www.bcb.gob.bo/webdocs/files_noticias/NP%2019-1%20SEPTIEMBRE.pdf",
        "date": "2026-09-01",
        "date_text": "01 de septiembre de 2026",
        "type": "notas-prensa"
    },
    {
        "title": "EL BANCO CENTRAL DE BOLIVIA RECIBE EN CUSTODIA ORO INCAUTADO",
        "pdf_url": "https://www.bcb.gob.bo/webdocs/files_noticias/NP%2016-11%20AGOSTO.pdf",
        "date": "2026-08-11",
        "date_text": "11 de agosto de 2026",
        "type": "notas-prensa"
    },
    {
        "title": "EL BANCO CENTRAL REALIZARÁ OPERACIONES DE COMPRA Y VENTA DE DÓLARES PARA MITIGAR FLUCTUACIONES EXTREMAS EN SU PRECIO",
        "pdf_url": "https://www.bcb.gob.bo/webdocs/files_noticias/NP%2015-30%20JULIO.pdf",
        "date": "2026-06-30",
        "date_text": "30 de julio de 2026",
        "type": "notas-prensa"
    },
    {
        "title": "AHORRISTAS PUEDEN RETIRAR DESDE HOY ENTRE $us 3.001 Y $us 5.000 DEL SISTEMA FINANCIERO",
        "pdf_url": "https://www.bcb.gob.bo/webdocs/files_noticias/NP%2017-14%20AGOSTO.pdf",
        "date": "2026-08-14",
        "date_text": "14 de agosto de 2026",
        "type": "notas-prensa"
    },
    {
        "title": "BANCO CENTRAL DE BOLIVIA ANUNCIA ACCIONES LEGALES ANTE INTENTOS DE ESTAFA",
        "pdf_url": "https://www.bcb.gob.bo/webdocs/files_noticias/NP%2014-28%20JULIO.pdf",
        "date": "2026-07-28",
        "date_text": "28 de julio de 2026",
        "type": "notas-prensa"
    },
    {
        "title": "ARRANCA LA SEGUNDA FASE DEL CRONOGRAMA DE DEVOLUCIÓN DE DEPÓSITOS EN DÓLARES",
        "pdf_url": "https://www.bcb.gob.bo/webdocs/files_noticias/NP11-15%20JULIO%20%281%29.pdf",
        "date": "2026-07-15",
        "date_text": "15 de julio de 2026",
        "type": "notas-prensa"
    },
    {
        "title": "PAGO POR PIGNORACIÓN DE ORO",
        "pdf_url": "https://www.bcb.gob.bo/webdocs/files_noticias/NP10-15%20JUNIO.pdf",
        "date": "2026-06-15",
        "date_text": "15 de junio de 2026",
        "type": "notas-prensa"
    },
     {
        "title": "BANCO CENTRAL DE BOLIVIA GARANTIZA EL RESGUARDO DE SUS RESERVAS Y ACTIVOS",
        "pdf_url": "https://www.bcb.gob.bo/webdocs/files_noticias/NP9-18%20MAYO.pdf",
        "date": "2026-05-18",
        "date_text": "18 de mayo de 2026",
        "type": "notas-prensa"
    },
      {
        "title": "EL BCB RATIFICA EL VALOR REFERENCIAL DEL DÓLAR ESTADOUNIDENSE",
        "pdf_url": "https://www.bcb.gob.bo/webdocs/files_noticias/NP8-7%20ABRIL.pdf",
        "date": "2026-04-07",
        "date_text": "07 de abril de 2026",
        "type": "notas-prensa"
    },
]

def load_existing_communications() -> tuple[list, set]:
    """Loads existing records and creates a set of existing PDF URLs for fast duplicate checking."""
    if JSON_PATH.exists():
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            communications = json.load(f)
    else:
        communications = []
    
    existing_urls = {item.get("pdf_url") for item in communications if item.get("pdf_url")}
    return communications, existing_urls

def extract_pdf_text(pdf_url: str) -> str:
    """Downloads a PDF and extracts its text content."""
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
        
        return "\n".join(text_pages).strip()
    except Exception as e:
        print(f"    ⚠ Error downloading or extracting PDF ({pdf_url}): {e}")
        return ""

def process_manual_entries():
    communications, existing_urls = load_existing_communications()
    added_count = 0

    print(f"Loaded {len(existing_urls)} existing documents from {JSON_PATH}")
    print("Checking manual entries...")

    for entry in MANUAL_ENTRIES:
        pdf_url = entry.get("pdf_url")
        
        # Check if the link is already included
        if pdf_url in existing_urls:
            print(f"  ⏭ Skipped (Already exists): {entry.get('title')} ({pdf_url})")
            continue

        print(f"  📥 Processing new manual entry: {entry.get('title')}...")
        full_text = extract_pdf_text(pdf_url)

        new_item = {
            "title": entry.get("title", "Untitled"),
            "pdf_url": pdf_url,
            "date_text": entry.get("date_text", ""),
            "date": entry.get("date", ""),
            "type": entry.get("type", "comunicados"),
            "full_text": full_text
        }

        communications.append(new_item)
        existing_urls.add(pdf_url)
        added_count += 1
        print(f"  ✓ Successfully added.")

    if added_count > 0:
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(communications, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Saved {added_count} new document(s) to {JSON_PATH}")
    else:
        print("\nNo new documents were added. All provided links already exist.")

if __name__ == "__main__":
    process_manual_entries()