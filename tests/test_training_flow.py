import torch
from torch.utils.data import DataLoader, Dataset
from src.training_utils import predict
from unittest.mock import MagicMock, Mock


class DummyDictDataset(Dataset):
    def __init__(self, num_samples=4, num_classes=3):
        self.num_samples = num_samples
        self.pixel_values = torch.randn(num_samples, 3, 224, 224)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "pixel_values": self.pixel_values[idx],
            "labels": self.labels[idx]
        }


def test_predict_flow():
    mock_model = MagicMock()
    mock_model.return_value = Mock(logits=torch.randn(2, 3))
    mock_model.eval = MagicMock()

    dummy_dataset = DummyDictDataset(num_samples=4)
    dataloader = DataLoader(dummy_dataset, batch_size=2)
    logits, labels = predict(mock_model, dataloader, device=torch.device("cpu"))
    mock_model.eval.assert_called_once()
    assert logits.shape == (4, 3)
    assert labels.shape == (4,)
    assert torch.equal(labels, dummy_dataset.labels) 
    assert mock_model.call_count == 2
    first_call_args = mock_model.call_args_list[0].kwargs
    assert "pixel_values" in first_call_args
    assert first_call_args['pixel_values'].shape == (2, 3, 224, 224)
