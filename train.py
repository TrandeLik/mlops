import argparse
import yaml
import logging
import torch
import mlflow
import mlflow.transformers
from pathlib import Path
import subprocess
from datasets import load_from_disk
from src.data_processing import create_transform_fn, apply_transform, create_dataloaders
from src.model_utils import get_model, get_image_processor
from src.utils import setup_logging, set_seed
from src.training_utils import run_training_loop, get_optimizer


def get_dvc_data_hash(stage_name: str, data_path: str) -> str:
    with open('dvc.lock') as f:
        lock_data = yaml.safe_load(f)
    current_stage = lock_data['stages'][stage_name]['outs']
    for stage_info in current_stage:
        if stage_info['path'] == data_path:
            return stage_info.get('md5')
    raise ValueError(f"Could not find hash for {data_path} in stage {stage_name}")

def get_commit_hash():
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip()[:8]


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def train_model(config_path: str):
    setup_logging()
    with open(config_path) as f:
        config = yaml.safe_load(f)
    set_seed(config['run']['seed'])

    mlflow.set_experiment("AI-Image-Detector")

    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")

    with mlflow.start_run() as run:
        logging.info("Starting model training...")
        data_hash = get_dvc_data_hash('prepare', config['data']['output_dir'])
        commit_hash = get_commit_hash()
        mlflow.set_tag("dvc_data_hash", data_hash)
        mlflow.set_tag("commit_hash", commit_hash)
        logging.info(f"DVC data hash: {data_hash}")
        logging.info(f"Commit hash: {commit_hash}")
        mlflow.log_artifact("dvc.lock")
        flat_config = flatten_dict(config)
        mlflow.log_params(flat_config)
        logging.info("Logged config parameters to MLflow.")
    
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
        mlflow.transformers.log_model(
            transformers_model={
                'model': trained_model,
                'image_processor': image_processor,
            },
            artifact_path='model',
            task='image-classification',
        )

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
