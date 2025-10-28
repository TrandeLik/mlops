# src/data_processing.py

import logging
from typing import Dict, Any, Callable
from PIL.Image import Image
from datasets import load_dataset, Dataset
from transformers import AutoImageProcessor
from torch.utils.data import DataLoader
import torch


def validate_dataset_sample(sample: Dict[str, Any], config: Dict[str, Any]):
    image_col = config['data']['image_column']
    label_col = config['data']['label_column']

    if image_col not in sample:
        raise ValueError(f"Required image column '{image_col}' not found in dataset.")
    if label_col not in sample:
        raise ValueError(f"Required label column '{label_col}' not found in dataset.")
    
    if not isinstance(sample[image_col], Image):
        raise TypeError(f"Image column '{image_col}' must be of type PIL.Image.Image, but got {type(sample[image_col])}.")
    if not isinstance(sample[label_col], int):
        raise TypeError(f"Label column '{label_col}' must be of type int, but got {type(sample[label_col])}.")


def load_and_validate_dataset(config: Dict[str, Any]) -> Dict[str, Dataset]:

    logging.info(f"Loading dataset '{config['data']['dataset_name']}'...")

    source_split_name = config['data_split']['source_split']
    full_dataset = load_dataset(config['data']['dataset_name'], split=source_split_name)
    
    logging.info("Validating dataset structure and types...")
    validate_dataset_sample(full_dataset[0], config)
    logging.info("Dataset validation successful.")

    logging.info(f"Splitting dataset into train/validation with validation size {config['data_split']['val_size']}...")
    split_dataset = full_dataset.train_test_split(
        test_size=config['data_split']['val_size'],
        seed=config['data_split']['shuffle_seed'],
        stratify_by_column=config['data']['label_column']
    )
    
    train_split = split_dataset['train']
    val_split = split_dataset['test']
    
    logging.info(f"Train split size: {len(train_split)}")
    logging.info(f"Validation split size: {len(val_split)}")

    return {"train": train_split, "val": val_split}


def create_transform_fn(image_processor: Any, config: Dict[str, Any]) -> Callable:
    image_col = config['data']['image_column']
    def transform(examples: Dict[str, Any]) -> Dict[str, Any]:
        images = [img.convert("RGB") for img in examples[image_col]]
        processed = image_processor(images, return_tensors="pt")
        examples["pixel_values"] = processed["pixel_values"]
        return examples
    return transform


def apply_transform(splits: Dict[str, Dataset], transform_fn: Callable) -> Dict[str, Dataset]:
    logging.info("Applying transformations to datasets...")
    transformed_splits = {}
    for split_name, data in splits.items():
        transformed_splits[split_name] = data.with_transform(transform_fn)
    return transformed_splits


def create_dataloaders(transformed_splits: Dict[str, Dataset], config: Dict[str, Any]) -> Dict[str, DataLoader]:
    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.tensor([x[config['data']['label_column']] for x in batch]),
        }

    train_loader = DataLoader(
        transformed_splits['train'],
        batch_size=config['training']['per_device_train_batch_size'],
        collate_fn=collate_fn,
        shuffle=True,
    )
    val_loader = DataLoader(
        transformed_splits['val'],
        batch_size=config['training']['per_device_eval_batch_size'],
        collate_fn=collate_fn,
    )
    logging.info("DataLoaders created successfully.")
    return {"train": train_loader, "val": val_loader}
