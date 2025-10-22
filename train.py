import argparse
import yaml
from src.utils import setup_logging, set_seed
from src.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train an image classification model.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the training config file."
    )
    args = parser.parse_args()

    setup_logging()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    set_seed(config['run']['seed'])
    
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
