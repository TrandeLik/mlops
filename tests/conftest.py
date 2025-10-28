import pytest
from PIL import Image
import numpy as np
from datasets import Dataset

@pytest.fixture(scope="session")
def mock_config():
    return {
        'data': {
            'image_column': 'image',
            'label_column': 'label',
        },
        'model': {
            'name': 'google/mobilenet_v2_1.0_224'
        }
    }

@pytest.fixture(scope="session")
def mock_image_dataset():
    def create_fake_image():
        return Image.fromarray(np.uint8(np.random.rand(10, 10, 3) * 255))

    data = {
        "image": [create_fake_image() for _ in range(5)],
        "label": [0, 1, 2, 0, 1],
    }
    return Dataset.from_dict(data)
