import argparse
import yaml
import logging
import torch
from pathlib import Path
from datasets import load_from_disk
from src.data_processing import create_transform_fn, apply_transform, create_dataloaders
from src.model_utils import get_model, get_image_processor
from src.utils import setup_logging, set_seed
from src.training_utils import run_training_loop, get_optimizer


def train_model(config_path: str):
    setup_logging()
    with open(config_path) as f:
        config = yaml.safe_load(f)
    set_seed(config['run']['seed'])

    logging.info("Starting model training...")
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    
    data_dir = Path(config['data']['output_dir'])
    train_dataset = load_from_disk(str(data_dir / "train.hf"))
    val_dataset = load_from_disk(str(data_dir / "val.hf"))
    splits = {"train": train_dataset, "val": val_dataset}

    image_processor = get_image_processor(config)
    transform_fn = create_transform_fn(image_processor, config)
    transformed_splits = apply_transform(splits, transform_fn)
    dataloaders = create_dataloaders(transformed_splits, config)
    
    labels_info = splits['train'].features[config['data']['label_column']]
    id2label = {i: label for i, label in enumerate(labels_info.names)}
    label2id = {label: i for i, label in enumerate(labels_info.names)}
    model = get_model(config, id2label, label2id).to(device)
    optimizer = get_optimizer(model, config)

    trained_model = run_training_loop(model, dataloaders['train'], dataloaders['val'], optimizer, config, device)

    model_output_dir = Path(config['training']['output_dir'])
    model_output_dir.mkdir(parents=True, exist_ok=True)
    trained_model.save_pretrained(model_output_dir)
    image_processor.save_pretrained(model_output_dir)
    logging.info(f"Model saved to {model_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config file", default="configs/train_config.yaml")
    args = parser.parse_args()
    train_model(args.config)
