import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from torchvision import transforms

MODEL_REPO = 'facebookresearch/dinov2'
MODEL_NAME = 'dinov2_vits14'

def load_dino_model():
    """
    Load a pre-trained DINO model from the specified path.
    """
    print(f"⏳ Loading {MODEL_NAME} from torch.hub...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load from Hub (downloads automatically to cache)
    model = torch.hub.load(MODEL_REPO, MODEL_NAME)
    model.to(device)
    model.eval()
    print(f"✅ {MODEL_NAME} Loaded Successfully!")
    return model, device

def get_transforms():
    """
    Define the image transformations for DINO model input.
    """
    dino_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # Resize to standard ViT input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return dino_transforms

def get_dino_embedding_from_path(model, image_path, transform, device):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        features = model.forward_features(img_tensor)
        # Use normalized CLS token as global embedding
        cls_token = features["x_norm_clstoken"]  # (1, 1, D)
        emb = cls_token.squeeze(0).squeeze(0)    # (D,)

    return emb.cpu()

def get_dino2_embeddings_for_location(image_files, model, transform, device):
    """
    image_files: dict like
      {
        'North': 'img_000000_north.jpg',
        'East':  'img_000000_east.jpg',
        'South': 'img_000000_south.jpg',
        'West':  'img_000000_west.jpg'
      }
    returns: (embs, directions)
      embs: (4, D) tensor in order [North, East, South, West]
      directions: list of direction strings in that order
    """
    directions = ["North", "East", "South", "West"]
    embs = []
    for d in directions:
        path = image_files[d]
        emb = get_dino_embedding_from_path(model, path, transform, device)
        embs.append(emb)

    embs = torch.stack(embs, dim=0)  # (4, D)
    return embs, directions
