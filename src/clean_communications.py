import json
from pathlib import Path

from src.data_collection import deduplicate_communications

RAW_PATH = Path("data/raw/raw_communications.json")
CLEAN_PATH = Path("data/processed/communications_clean.json")

def clean_communications() -> Path:
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    clean = deduplicate_communications(raw)
    clean.sort(key=lambda x: x["date"])

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLEAN_PATH, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)

    print(f"Raw records: {len(raw)} -> Unique documents: {len(clean)}")
    return CLEAN_PATH

if __name__ == "__main__":
    clean_communications()