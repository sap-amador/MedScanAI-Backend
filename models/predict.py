import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from gradcam_utils import generate_heatmap

# Dummy model mapping (replace with real models as needed)
MODEL_PATHS = {
    "chest_xray": "models/chest_xray_model.pt",
    "brain_ct": "models/brain_ct_model.pt",
    "mammogram": "models/mammo_model.pt",
    "msk_xray": "models/msk_model.pt",
    "ultrasound": "models/ultrasound_model.pt"
}

def load_model(modality):
    model_path = MODEL_PATHS.get(modality)
    if not model_path or not os.path.exists(model_path):
        raise ValueError(f"Model for {modality} not found.")
    
    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    return model

def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_bytes).convert("RGB")
    return transform(image).unsqueeze(0)

def predict_image_from_modality(modality, image_file):
    model = load_model(modality)
    image_tensor = transform_image(image_file)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    # Optional: Class labels
    labels = ["Normal", "Pneumonia", "COVID", "TB"] if modality == "chest_xray" else ["Class 0", "Class 1"]
    predicted_class = labels[predicted.item()] if predicted.item() < len(labels) else "Unknown"

    # Grad-CAM heatmap
    heatmap_path = generate_heatmap(model, image_tensor, modality)

    return {
        "prediction": predicted_class,
        "heatmap_path": heatmap_path
    }
