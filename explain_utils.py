import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

def generate_gradcam(model, img: Image.Image, target_layer: str, class_idx: int):
    """
    Generate Grad-CAM heatmap for a single image.
    Args:
        model: torch.nn.Module (classification model)
        img: PIL Image
        target_layer: str, e.g. "features.6" for EfficientNet-B0
        class_idx: int, class index to explain
    Returns:
        heatmap overlaid on original image (PIL Image)
    """
    model.eval()
    device = next(model.parameters()).device

    # Transform image like in training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Hook the target layer
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    layer = dict([*model.named_modules()])[target_layer]
    fh = layer.register_forward_hook(forward_hook)
    bh = layer.register_backward_hook(backward_hook)

    # Forward + backward
    output = model(input_tensor)
    score = output[:, class_idx]
    model.zero_grad()
    score.backward()

    # Compute Grad-CAM
    grads = gradients["value"]      # [N, C, H, W]
    acts = activations["value"]     # [N, C, H, W]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    # Resize to image size
    cam = cv2.resize(cam, (img.size[0], img.size[1]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = np.array(img) * 0.5 + heatmap * 0.5
    overlay = np.uint8(overlay)

    # Clean hooks
    fh.remove()
    bh.remove()

    return Image.fromarray(overlay)
