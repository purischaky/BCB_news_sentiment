from unittest.mock import MagicMock, patch
import pytest
from src.signal_classifier import BCBSignalClassifier

def test_classify_statement_empty_text():
    """Test behavior with empty or missing text input."""
    classifier = BCBSignalClassifier.__new__(BCBSignalClassifier)
    result = classifier.classify_statement("")
    
    assert result["intensity_score"] == 0.0
    assert result["is_hawkish"] is False
    assert result["stance_label"] == "comunicado administrativo o rutinario"

@patch("src.signal_classifier.pipeline")
def test_classify_statement_hawkish_signal(mock_pipeline):
    """Test correct identification of a Hawkish / restrictive statement."""
    mock_classifier_func = MagicMock()
    mock_classifier_func.return_value = {
        "labels": ["restricción de divisas o control de cambios", "estabilidad de reservas y liquidez"],
        "scores": [0.85, 0.15]
    }
    mock_pipeline.return_value = mock_classifier_func

    classifier = BCBSignalClassifier()
    sample_text = "El banco central anuncia restricciones mas estrictas para el acceso a divisas."
    
    result = classifier.classify_statement(sample_text)

    assert result["stance_label"] == "restricción de divisas o control de cambios"
    assert result["intensity_score"] == 0.85
    assert result["is_hawkish"] is True