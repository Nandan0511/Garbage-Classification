import os
from typing import List, Tuple
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
CHECKPOINT_PATH = r"C:\Users\nanda\OneDrive\Desktop\Garbage Classifier\RECYCLIZER\Model\best_model_2.pth"
FALLBACK_CLASSES = ["Hazardous-samples", "Non-Recycable-samples", "Organic-samples", "Recyclable-samples"]
DATA_DIR_FOR_CLASSES = "data1/train"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_classes() -> List[str]:
    """Try to find class names from txt file, dataset dir, or fallback list."""
    if os.path.isfile("class_names.txt"):
        with open("class_names.txt", "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f.readlines() if ln.strip()]
        if names:
            return names
    if os.path.isdir(DATA_DIR_FOR_CLASSES):
        sub = sorted([d for d in os.listdir(DATA_DIR_FOR_CLASSES)
                      if os.path.isdir(os.path.join(DATA_DIR_FOR_CLASSES, d))])
        if sub:
            return sub
    return FALLBACK_CLASSES


def load_model(num_classes: int):
    """Load EfficientNet-B0 with custom head."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )
    model.to(DEVICE)

    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def build_transform():
    """Preprocessing pipeline."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def predict_one(model: torch.nn.Module, img_pil: Image.Image,
                class_names: List[str], topk: int = 4) -> Tuple[List[str], List[float]]:
    """Run inference on a single PIL image."""
    tfm = build_transform()
    x = tfm(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().squeeze(0)
    vals, idxs = torch.topk(probs, k=min(topk, len(class_names)))
    top_classes = [class_names[i] for i in idxs.tolist()]
    top_probs = vals.tolist()
    return top_classes, top_probs
