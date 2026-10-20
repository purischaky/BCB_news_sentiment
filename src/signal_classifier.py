import json
from pathlib import Path
from transformers import pipeline

# Configuration & Paths 
DATA_RAW_PATH = Path("data/raw/raw_communications.json")
DATA_PROCESSED_DIR = Path("data/processed")
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

class BCBSignalClassifier:
    """
    NLP Signal Classifier for Central Bank Communications.
    Uses Zero-Shot Classification to quantify policy stance and intensity.
    """

    def __init__(self, model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"):
        print(f"Loading Hugging Face Zero-Shot Model: {model_name}...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model=model_name
        )
        
        self.candidate_labels = [
            "restricción de divisas o control de cambios",
            "ajuste de política monetaria o régimen cambiario",
            "estabilidad de reservas y liquidez",
            "comunicado administrativo o rutinario"
        ]

    def classify_statement(self, text: str) -> dict:
        """
        Evaluates document text to assign policy stance, confidence score, 
        and a boolean flag indicating a strong hawkish/risk signal.
        """
        if not text or len(text.strip()) < 50:
            return {
                "stance_label": "comunicado administrativo o rutinario",
                "intensity_score": 0.0,
                "is_hawkish": False
            }

        # Truncate to first 800 characters as a first-pass character safeguard
        summary_text = text[:800]

        try:
            # Enforce explicit token truncation at the pipeline level to prevent DeBERTa attention crashes
            result = self.classifier(
                summary_text, 
                candidate_labels=self.candidate_labels,
                truncation=True,
                max_length=512
            )
            
            top_label = result["labels"][0]
            top_score = float(result["scores"][0])

            is_hawkish = (top_score >= 0.50) and (
                "restricción" in top_label or "ajuste" in top_label
            )

            return {
                "stance_label": top_label,
                "intensity_score": round(top_score, 4),
                "is_hawkish": is_hawkish
            }

        except Exception as e:
            print(f"    ⚠ Warning: Failed to classify document chunk: {e}")
            return {
                "stance_label": "comunicado administrativo o rutinario",
                "intensity_score": 0.0,
                "is_hawkish": False
            }

def process_raw_communications(
    input_path: Path = DATA_RAW_PATH, 
    output_filename: str = "processed_signals.json"
) -> Path:
    """
    Reads raw communications JSON, runs NLP classification, 
    and saves the enriched dataset to data/processed/.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at {input_path}. Run src/data_collection.py first.")

    print(f"Reading raw data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    classifier = BCBSignalClassifier()
    processed_records = []

    print(f"\nProcessing {len(raw_data)} communications with NLP classifier...")
    for idx, item in enumerate(raw_data, start=1):
        full_text = item.get("full_text", "")
        signal = classifier.classify_statement(full_text)

        record = {
            "title": item["title"],
            "date": item["date"],
            "type": item["type"],
            "pdf_url": item["pdf_url"],
            "stance_label": signal["stance_label"],
            "intensity_score": signal["intensity_score"],
            "is_hawkish": signal["is_hawkish"]
        }
        processed_records.append(record)
        print(f"  [{idx}/{len(raw_data)}] {item['date']} -> {signal['stance_label']} (Score: {signal['intensity_score']})")

    output_path = DATA_PROCESSED_DIR / output_filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_records, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Enriched dataset successfully saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    process_raw_communications()