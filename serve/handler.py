import logging
import torch
from ts.torch_handler.base_handler import BaseHandler
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)

class AIImageDetectorHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model = AutoModelForImageClassification.from_pretrained(model_dir).to(self.device).eval()
        self.image_processor = AutoImageProcessor.from_pretrained(model_dir)
        self.id2label = self.model.config.id2label

        logger.info("Model and processor loaded successfully.")
        self.initialized = True

    def preprocess(self, data):
        images = []
        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                image = base64.b64decode(image)
            image = Image.open(io.BytesIO(image)).convert("RGB")
            images.append(image)
        
        inputs = self.image_processor(images, return_tensors="pt")
        return inputs.to(self.device)

    def inference(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs).logits
        return outputs

    def postprocess(self, inference_output):
        probabilities = torch.nn.functional.softmax(inference_output, dim=1)
        top_probs, top_indices = torch.max(probabilities, dim=1)
        
        predictions = []
        for i in range(len(top_probs)):
            class_id = top_indices[i].item()
            predictions.append({
                "label": self.id2label[class_id],
                "confidence": top_probs[i].item()
            })
        return predictions
