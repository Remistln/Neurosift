import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelPredictor:
    def __init__(self, model_path="models/neurosift_resnet18.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ["T1", "T2", "FLAIR"]
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _load_model(self, path):
        if not os.path.exists(path):
            logger.warning(f"Model not found at {path}")
            return None
            
        model = models.resnet18(weights=None) # Structure only
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(self.classes))
        
        try:
            model.load_state_dict(torch.load(path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def predict(self, image_path):
        if not self.model:
            return None

        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                class_idx = predicted.item()
                label = self.classes[class_idx]
                score = confidence.item()
                
                return {"label": label, "confidence": score}
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
