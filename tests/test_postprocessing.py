import pytest
import torch
from src.postprocessing import convert_logits_to_predictions


@pytest.fixture
def sample_logits():
    return torch.tensor([
        [0.1, 2.5], 
        [3.0, 0.5]
    ])


@pytest.fixture
def sample_id2label():
    return {0: "Real", 1: "AI", 2: "Deepfake"}

def test_convert_logits_to_predictions_format(sample_logits, sample_id2label):
    preds = convert_logits_to_predictions(sample_logits, sample_id2label)
    
    assert isinstance(preds, list)
    assert len(preds) == 2
    assert isinstance(preds[0], dict)
    assert "label" in preds[0]
    assert "confidence" in preds[0]


def test_convert_logits_to_predictions_values(sample_logits, sample_id2label):
    preds = convert_logits_to_predictions(sample_logits, sample_id2label)

    assert preds[0]["label"] == "AI"
    assert isinstance(preds[0]["confidence"], float)
    assert 0.0 <= preds[0]["confidence"] <= 1.0

    assert preds[1]["label"] == "Real"
    assert isinstance(preds[1]["confidence"], float)


def test_convert_logits_to_predictions_invalid_input():
    with pytest.raises(ValueError, match="Logits must be a 2D torch.Tensor"):
        convert_logits_to_predictions(torch.tensor([1, 2, 3]), {})

