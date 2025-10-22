import os
import logging
import torch
from transformers import PreTrainedModel

from . import data_utils, model_utils, training_utils


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(
            config['training']['device'] if torch.cuda.is_available() else "cpu"
        )
        logging.info(f"Pipeline initialized on device: {self.device}")

    def _prepare_data(self) -> tuple[dict, dict, dict]:
        dataloaders, id2label, label2id = data_utils.get_dataloaders(self.config)
        return dataloaders, id2label, label2id

    def train(self):
        dataloaders, id2label, label2id = self._prepare_data()

        model = model_utils.get_model(self.config, id2label, label2id)
        model.to(self.device)
        optimizer = training_utils.get_optimizer(model, self.config)

        trained_model = training_utils.run_training_loop(
            model, 
            dataloaders['train'], 
            dataloaders['val'],
            optimizer, 
            self.config,
            self.device,
        )
        self._save_model(trained_model)
        logging.info("Training pipeline finished successfully!")

    def _save_model(self, model: PreTrainedModel):
        output_dir = self.config['training']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        logging.info(f"Saving model to {output_dir}...")
        model.save_pretrained(output_dir)
        
        image_processor = model_utils.get_image_processor(self.config)
        image_processor.save_pretrained(output_dir)
