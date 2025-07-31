import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io

from .utils import generate_gradcam

# Load CheXNet (DenseNet121 trained on ChestX-ray14)
def load_model():
    model = models.densenet121(pretrained=True)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 14)  # 14 NIH labels
    model.eval()
    target_layer = model.features[-1]
    return model, target_layer

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

def predict(image_bytes):
    model, target_layer = load_model()
    input_tensor = transform_image(image_bytes)
    outputs = model(input_tensor)
    _, predicted = outputs.max(1)
    return predicted.item(), input_tensor, model, target_layer
