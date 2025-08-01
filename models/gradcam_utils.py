import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_heatmap(model, image_tensor, modality):
    # This is a simplified Grad-CAM placeholder
    cam = torch.mean(image_tensor[0], dim=0).numpy()

    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.axis('off')
    os.makedirs("static/heatmaps", exist_ok=True)
    heatmap_file = f"static/heatmaps/{modality}_heatmap.png"
    plt.savefig(heatmap_file, bbox_inches='tight')
    plt.close()

    return f"/{heatmap_file}"
