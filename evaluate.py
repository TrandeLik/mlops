import argparse
import yaml
import logging
import json
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoModelForImageClassification
from src.data_processing import create_transform_fn, apply_transform, create_dataloaders
from src.model_utils import get_image_processor
from src.utils import setup_logging
from src.training_utils import evaluate


def evaluate_model(config_path: str):
    setup_logging()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logging.info("Evaluating model...")
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    
    data_dir = Path(config['data']['output_dir'])
    val_dataset = load_from_disk(str(data_dir / "val.hf"))
    model_dir = Path(config['training']['output_dir'])
    model = AutoModelForImageClassification.from_pretrained(model_dir).to(device)
    
    image_processor = get_image_processor(config)
    transform_fn = create_transform_fn(image_processor, config)
    transformed_val = val_dataset.with_transform(transform_fn)
    val_loader = create_dataloaders({"val": transformed_val, "train": transformed_val}, config)['val']
    
    accuracy, report_str = evaluate(model, val_loader, device)

    metrics_path = Path(config['evaluation']['metrics_file'])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_dict = {}
    lines = report_str.strip().split('\n')
    for line in lines[2:-4]:
        if line == '':
            continue
        parts = line.split()
        class_name = parts[0]
        report_dict[class_name] = {
            "precision": float(parts[1]),
            "recall": float(parts[2]),
            "f1-score": float(parts[3]),
            "support": int(parts[4]),
        }
    
    metrics = {"overall_accuracy": accuracy, "class_metrics": report_dict}
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    logging.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config file", default="configs/train_config.yaml")
    args = parser.parse_args()
    evaluate_model(args.config)
