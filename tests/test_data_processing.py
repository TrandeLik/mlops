import pytest
from src.data_processing import validate_dataset_sample
from PIL import Image
import torch
from torch.utils.data import DataLoader
from src.data_processing import create_transform_fn, apply_transform, create_dataloaders


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


def test_create_transform_fn(mock_image_processor, mock_config, mock_image_dataset):
    transform_fn = create_transform_fn(mock_image_processor, mock_config)
    batch_examples = mock_image_dataset[:2]
    assert isinstance(batch_examples['image'], list)
    assert len(batch_examples['image']) == 2
    transformed_batch = transform_fn(batch_examples)
    assert "pixel_values" in transformed_batch
    mock_image_processor.assert_called_once()
    assert isinstance(mock_image_processor.call_args[0][0][0], Image.Image)


def test_create_dataloaders(mock_config, mock_image_dataset):
    def fake_transform(examples):
        examples["pixel_values"] = [torch.randn(3, 224, 224) for _ in examples['image']]
        return examples
    
    transformed_dataset = mock_image_dataset.with_transform(fake_transform)
    splits = {"train": transformed_dataset, "val": transformed_dataset}
    
    dataloaders = create_dataloaders(splits, mock_config)
    
    assert "train" in dataloaders and "val" in dataloaders
    assert isinstance(dataloaders['train'], DataLoader)
    
    batch = next(iter(dataloaders['train']))
    assert "pixel_values" in batch
    assert "labels" in batch
    assert batch['pixel_values'].shape[0] == mock_config['training']['per_device_train_batch_size']
    assert batch['labels'].dtype == torch.int64
