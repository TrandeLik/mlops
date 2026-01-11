import argparse
import logging
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageClassification, AutoImageProcessor

from src.postprocessing import convert_logits_to_predictions
from src.utils import setup_logging


MODEL_DIR = Path("./models/final_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_single_image(model, image_processor, image_path: Path) -> dict:
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = image_processor(img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**inputs).logits
        prediction = convert_logits_to_predictions(logits, model.config.id2label)[0]
        return prediction
    except Exception as e:
        logging.error(f"Failed to process {image_path}: {e}")
        return {"label": "error", "confidence": 0.0}


def batch_predict(input_path: str, output_path: str):
    setup_logging()
    logging.info(f"Using device: {DEVICE}")
    
    if not MODEL_DIR.exists():
        logging.error(f"Model directory not found at {MODEL_DIR}. Did you run 'dvc pull'?")
        return

    logging.info(f"Loading model from {MODEL_DIR}...")
    model = AutoModelForImageClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model.eval()

    input_dir = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpeg"))
    logging.info(f"Found {len(image_files)} images to process in {input_dir}.")
    
    results = []
    for image_path in tqdm(image_files, desc="Processing images"):
        prediction = predict_single_image(model, image_processor, image_path)
        results.append({
            "filename": image_path.name,
            "predicted_label": prediction['label'],
            "confidence": prediction['confidence']
        })

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    logging.info(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch predict on a folder of images.")
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the directory with input images."
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the output CSV predictions."
    )
    args = parser.parse_args()
    
    batch_predict(args.input_path, args.output_path)
