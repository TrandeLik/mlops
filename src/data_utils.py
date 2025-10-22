import logging
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor


def get_dataloaders(config: dict):
    logging.info(f"Loading dataset '{config['data']['dataset_name']}'...")

    source_split_name = config['data_split']['source_split']
    dataset = load_dataset(config['data']['dataset_name'], split=source_split_name)
    split_dataset = dataset.train_test_split(
        test_size=config['data_split']['val_size'],
        seed=config['data_split']['shuffle_seed'],
        stratify_by_column=config['data']['label_column'] # Важно для сохранения баланса классов
    )
    train_split = split_dataset['train']
    val_split = split_dataset['test']
    logging.info(f"Train split size: {len(train_split)}")
    logging.info(f"Validation split size: {len(val_split)}")

    labels = train_split.features[config['data']['label_column']].names
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    logging.info(f"Found labels: {labels}")
    image_processor = AutoImageProcessor.from_pretrained(config['model']['name'])

    def transform(examples):
        images = [img.convert("RGB") for img in examples[config['data']['image_column']]]
        examples["pixel_values"] = image_processor(images, return_tensors="pt")["pixel_values"]
        return examples

    train_dataset = train_split.with_transform(transform)
    val_dataset = val_split.with_transform(transform)

    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.tensor([x[config['data']['label_column']] for x in batch]),
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['per_device_train_batch_size'],
        collate_fn=collate_fn,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['per_device_eval_batch_size'],
        collate_fn=collate_fn,
    )

    logging.info("DataLoaders created successfully.")
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
    }
    return dataloaders, id2label, label2id

