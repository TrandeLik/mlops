import logging
from transformers import AutoModelForImageClassification, AutoImageProcessor


def get_model(config: dict, id2label: dict, label2id: dict):
    model_name = config['model']['name']
    logging.info(f"Loading pre-trained model '{model_name}'...")
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    logging.info("Model loaded and configured for classification.")
    return model


def get_image_processor(config: dict):
    model_name = config['model']['name']
    logging.info(f"Loading image processor for '{model_name}'...")
    return AutoImageProcessor.from_pretrained(model_name)
