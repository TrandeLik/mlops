from src.model_utils import get_model
from unittest.mock import MagicMock

def test_get_model(mocker, mock_config):
    mock_from_pretrained = mocker.patch(
        "src.model_utils.AutoModelForImageClassification.from_pretrained"
    )
    mock_model = MagicMock()
    mock_from_pretrained.return_value = mock_model

    id2label = {0: "Real", 1: "Fake"}
    label2id = {"Real": 0, "Fake": 1}

    model = get_model(mock_config, id2label, label2id)

    mock_from_pretrained.assert_called_once_with(
        mock_config['model']['name'],
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    
    assert model is mock_model
