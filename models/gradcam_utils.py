import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_heatmap(model, image_tensor, modality):
    # This example assumes model.features exists
    def forward_hook(module, input, output):
        activations.append(output)

    activations = []
    hook = model.features.register_forward_hook(forward_hook)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1).item()

    grads = torch.autograd.grad(outputs=output[0, pred_class], inputs=image_tensor)[0]

    heatmap = grads[0].mean(dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    plt.imshow(heatmap, cmap='jet')
    heatmap_path = f"static/heatmap_{modality}.png"
    os.makedirs("static", exist_ok=True)
    plt.axis('off')
    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
    hook.remove()
    return heatmap_path
