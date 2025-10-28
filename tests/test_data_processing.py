import pytest
from src.data_processing import validate_dataset_sample
from PIL import Image


def test_validate_dataset_sample_success(mock_config, mock_image_dataset):
    sample = mock_image_dataset[0]
    try:
        validate_dataset_sample(sample, mock_config)
    except ValueError:
        pytest.fail("Validation unexpectedly failed on correct data.")


def test_validate_dataset_sample_missing_image_col(mock_config):
    bad_sample = {'label': 0}
    with pytest.raises(ValueError, match="Required image column 'image' not found"):
        validate_dataset_sample(bad_sample, mock_config)


def test_validate_dataset_sample_wrong_image_type(mock_config):
    bad_sample = {'image': "not_an_image", 'label': 0}
    with pytest.raises(TypeError, match="must be of type PIL.Image.Image"):
        validate_dataset_sample(bad_sample, mock_config)


def test_validate_dataset_sample_wrong_label_type(mock_config):
    bad_sample = {'image': Image.new('RGB', (10, 10)), 'label': 'not_an_int'}
    with pytest.raises(TypeError, match="must be of type int"):
        validate_dataset_sample(bad_sample, mock_config)
