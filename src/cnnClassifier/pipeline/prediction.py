import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from pathlib import Path
import os



class PredictionPipeline:
    def __init__(self, filename: str):
        # Define the transformation: resize, center-crop, convert to tensor, and normalize
        self.transform = transforms.Compose([
            transforms.Resize(size=64),
            transforms.CenterCrop(size=180),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.device = torch.device("cpu")
        self.model = self.load_model(os.path.join("model", "model.pth"), self.device)  # Replace with your model path
        self.filename = filename
    
    @staticmethod
    def load_model(path: Path, device) -> torch.nn.Module:
        checkpoint = torch.load(path, map_location=device)
        model = models.vgg16(pretrained=True)
        model.eval()  # Set the model to evaluation mode
        model.classifier = checkpoint['classifier']
        # Load in the state dict
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        return model

    def preprocess_image(self):
        # Load the image and apply the transformation
        image = Image.open(self.filename)
        image = self.transform(image).unsqueeze(0)  # Add a batch dimension
        return image

    def predict(self):
        # Preprocess the image
        image = self.preprocess_image()
        image = image.to(self.device)
        # Predict the class
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
        
        print(f'Predicted class index: {predicted.item()}')
        
        if predicted.item() == 1:
            return [{"image": "Tumor"}]
        else:
            return [{"image": "Normal"}]