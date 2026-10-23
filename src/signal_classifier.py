import json
from pathlib import Path
from transformers import pipeline

# Configuration & Paths 
INPUT_PATH = Path("data/processed/communications_clean.json")
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
        Evaluates document text across chunks to capture policy stance and intensity 
        across the complete document without truncation loss.
        """
        if not text or len(text.strip()) < 50:
            return {
                "stance_label": "comunicado administrativo o rutinario",
                "intensity_score": 0.0,
                "is_hawkish": False
            }

        # Split text into chunks of ~1,000 characters to fit safely within token limits
        chunk_size = 1000
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        
        max_intensity = 0.0
        best_label = "comunicado administrativo o rutinario"
        highest_hawkish = False

        for chunk in chunks:
            if len(chunk.strip()) < 30:
                continue

            try:
                result = self.classifier(
                    chunk, 
                    candidate_labels=self.candidate_labels,
                    truncation=True,
                    max_length=512
                )
                
                top_label = result["labels"][0]
                top_score = float(result["scores"][0])

                is_hawkish = (top_score >= 0.50) and (
                    "restricción" in top_label or "ajuste" in top_label
                )

                # Track the strongest policy signal found across any section of the PDF
                if top_score > max_intensity:
                    max_intensity = top_score
                    best_label = top_label
                    highest_hawkish = is_hawkish

            except Exception as e:
                # Silently skip malformed chunks and continue processing the rest
                continue

        return {
            "stance_label": best_label,
            "intensity_score": round(max_intensity, 4),
            "is_hawkish": highest_hawkish
        }

def process_raw_communications(
    input_path: Path = INPUT_PATH, 
    output_filename: str = "processed_signals.json"
) -> Path:
    """
    Reads raw communications JSON, runs NLP classification, 
    and saves the enriched dataset to data/processed/.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at {input_path}. Run python -m src.clean_communications first.")

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