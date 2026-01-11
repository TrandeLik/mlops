import argparse
import yaml
import logging
from pathlib import Path
import sys
sys.path.append('..')
from src.data_processing import load_and_validate_dataset
from src.utils import setup_logging


def prepare_data(config_path: str):
    setup_logging()
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logging.info("Preparing data...")
    splits = load_and_validate_dataset(config)

    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, dataset in splits.items():
        output_path = output_dir / f"{split_name}.hf"
        dataset.save_to_disk(str(output_path))
        logging.info(f"Saved {split_name} split to {output_path}")

    logging.info("Data preparation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config file", default="configs/train_config.yaml")
    args = parser.parse_args()
    prepare_data(args.config)
