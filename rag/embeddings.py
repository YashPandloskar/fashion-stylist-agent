"""
FashionCLIP embedding wrapper.
Uses Marqo's fashion-specific CLIP model (Apache 2.0) — significantly
better than generic CLIP for clothing retrieval tasks.

Model: patrickjohncyh/fashion-clip (HuggingFace)
"""

from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
from typing import Union
import numpy as np

MODEL_NAME = "patrickjohncyh/fashion-clip"

# Module-level cache — load once, reuse across calls
_model = None
_processor = None


def _load_model():
    global _model, _processor
    if _model is None:
        print(f"Loading FashionCLIP model ({MODEL_NAME})...")
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        _model = CLIPModel.from_pretrained(MODEL_NAME)

        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        _model.eval()
        print(f"FashionCLIP loaded on {device}.")
    return _model, _processor


def embed_text(text: str) -> list[float]:
    """Embed a text string into a 512-dim FashionCLIP vector."""
    model, processor = _load_model()
    device = next(model.parameters()).device

    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)  # L2 normalise

    return features.squeeze().cpu().numpy().tolist()


def embed_image(image: Union[str, Image.Image]) -> list[float]:
    """Embed an image (path or PIL Image) into a 512-dim FashionCLIP vector."""
    model, processor = _load_model()
    device = next(model.parameters()).device

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.squeeze().cpu().numpy().tolist()
