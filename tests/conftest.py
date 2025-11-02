import pytest
from PIL import Image
import numpy as np
from datasets import Dataset, Features, ClassLabel, Image as HFImage
from unittest.mock import MagicMock
import torch

@pytest.fixture(scope="session")
def mock_config():
    return {
        'data': {
            'image_column': 'image',
            'label_column': 'label',
        },
        'data_split': {
            'val_size': 0.2
        },
        'model': {
            'name': 'google/mobilenet_v2_1.0_224'
        },
        'training': {
            'per_device_train_batch_size': 2,
            'per_device_eval_batch_size': 2,
            'device': 'cuda:0',
        }
    }

@pytest.fixture(scope="session")
def mock_image_dataset():
    def create_fake_image():
        return Image.fromarray(np.uint8(np.random.rand(30, 30, 3) * 255))
    num_classes = 3
    num_samples = 10
    features = Features({
        'image': HFImage(),
        'label': ClassLabel(num_classes=num_classes, names=[f'class_{i}' for i in range(num_classes)])
    })
    data = {
        "image": [create_fake_image() for _ in range(num_samples)],
        "label": np.random.randint(0, num_classes, num_samples).tolist(),
    }
    ds = Dataset.from_dict(data, features=features)
    return ds

@pytest.fixture
def mock_image_processor():
    mock = MagicMock()
    mock.return_value = {"pixel_values": torch.randn(3, 224, 224)}
    return mock