import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def generate_gradcam(model, image_tensor, target_layer, output_path="static/heatmap.png"):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    hook_f = target_layer.register_forward_hook(forward_hook)
    hook_b = target_layer.register_backward_hook(backward_hook)

    image_tensor.requires_grad_()
    output = model(image_tensor)
    class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()

    grad = gradients[0][0]
    act = activations[0][0]

    weights = grad.mean(dim=(1, 2))
    cam = torch.sum(weights[:, None, None] * act, dim=0).detach().numpy()

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.imshow(cam, cmap='jet')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

    hook_f.remove()
    hook_b.remove()

    return output_path
