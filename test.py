import argparse
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Mitigate macOS OpenMP duplicate runtime issues
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from config import BATCH_SIZE
from data_loader import create_directional_test_dataloader
from model import StreetCLIPMultiTask
from transformers import CLIPProcessor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict(model, dataloader, device):
    """Run inference and collect top-5 state predictions, GPS preds, and sample ids."""
    model.eval()
    all_top5 = []
    all_gps = []
    all_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            pixel_values = batch['pixel_values'].to(device)

            state_logits, gps_pred = model(pixel_values)

            top5 = torch.topk(state_logits, k=5, dim=1).indices  # [B, 5]

            all_top5.append(top5.cpu().numpy())
            all_gps.append(gps_pred.cpu().numpy())

            if 'sample_id' in batch:
                all_ids.append(batch['sample_id'])
            elif 'image_id' in batch:
                all_ids.append(batch['image_id'])

    all_top5 = np.concatenate(all_top5, axis=0)
    all_gps = np.concatenate(all_gps, axis=0)

    # Normalize ids to a flat numpy array of strings
    if all_ids:
        first = all_ids[0]
        if torch.is_tensor(first):
            all_ids = torch.cat(all_ids).cpu().numpy()
        else:
            all_ids = np.concatenate([np.array(ids) for ids in all_ids], axis=0)
    else:
        all_ids = np.arange(len(all_top5))

    return all_top5, all_gps, all_ids


def save_predictions_to_csv(top5_preds, gps_preds, sample_ids, csv_path):
    """Write predictions to CSV with specified columns (top-1 state only)."""
    assert len(top5_preds) == len(gps_preds) == len(sample_ids)

    # Ensure ids are strings
    sample_ids = [str(sid) for sid in sample_ids]

    # Build image filenames from id
    img_cols = {
        "image_north": [f"image_{sid}_north.jpg" for sid in sample_ids],
        "image_east": [f"image_{sid}_east.jpg" for sid in sample_ids],
        "image_south": [f"image_{sid}_south.jpg" for sid in sample_ids],
        "image_west": [f"image_{sid}_west.jpg" for sid in sample_ids],
    }

    df = pd.DataFrame({
        "sample_id": sample_ids,
        **img_cols,
        "predicted_state_idx_1": top5_preds[:, 0],
        "predicted_latitude": gps_preds[:, 0],
        "predicted_longitude": gps_preds[:, 1],
    })

    df.to_csv(csv_path, index=False)


def create_single_sample_dataloader(image_dir, image_id="000000"):
    """Create a one-sample dataloader for a specific id with 4 directional images."""
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
    directions = ["north", "east", "south", "west"]
    images = []

    for direction in directions:
        # Support both 'image_' and 'img_' filename prefixes
        candidates = [
            os.path.join(image_dir, f"image_{image_id}_{direction}.jpg"),
            os.path.join(image_dir, f"img_{image_id}_{direction}.jpg"),
        ]
        image = None
        img_path = None
        for p in candidates:
            img_path = p
            image = cv2.imread(p)
            if image is not None:
                break
        if image is None:
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    inputs = processor(images=images, return_tensors="pt", padding=True)
    sample = {"pixel_values": inputs["pixel_values"], "image_id": image_id}
    dataset = [sample]
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return loader, processor


def inspect_single_image(image_path):
    """Open a single image and print basic info to stdout."""
    if not os.path.exists(image_path):
        print(f"[inspect] File not found: {image_path}")
        return False
    img = cv2.imread(image_path)
    if img is None:
        print(f"[inspect] Failed to read image: {image_path}")
        return False
    h, w, c = img.shape
    print(f"[inspect] Opened {image_path}: shape={h}x{w}x{c}, dtype={img.dtype}")
    print(f"[inspect] Min/Max: {img.min()} / {img.max()}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference and export predictions.")
    parser.add_argument("--image_dir", type=str, default="./kaggle_dataset/test_images", help="Path to test images directory.")
    parser.add_argument("--model_path", type=str, default="best_streetclip_model.pth", help="Path to trained model weights.")
    parser.add_argument("--output_npz", type=str, default="predictions.npz", help="Path to save raw predictions (npz).")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Path to save predictions CSV.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for inference.")
    parser.add_argument("--single_id", type=str, default=None, help="If set, run a small test on this image id (e.g., 000000).")
    parser.add_argument("--inspect_image", type=str, default=None, help="If set, only inspect this image path and exit.")
    args = parser.parse_args()

    # Simple inspection mode: open a single image and exit
    if args.inspect_image:
        ok = inspect_single_image(args.inspect_image)
        raise SystemExit(0 if ok else 1)

    # Create dataloader (full set or single-id smoke test)
    if args.single_id:
        test_loader, _ = create_single_sample_dataloader(args.image_dir, image_id=args.single_id)
    else:
        test_loader, _ = create_directional_test_dataloader(args.image_dir, batch_size=args.batch_size)

    # Load model
    model = StreetCLIPMultiTask(num_states=50)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.to(DEVICE)

    # Generate predictions
    top5_preds, gps_preds, sample_ids = predict(model, test_loader, DEVICE)

    # Save predictions to NPZ
    np.savez_compressed(args.output_npz, top5_predictions=top5_preds, gps_predictions=gps_preds, sample_ids=sample_ids)
    print(f"Predictions saved to {args.output_npz}")

    # Save predictions to CSV
    save_predictions_to_csv(top5_preds, gps_preds, sample_ids, args.output_csv)
    print(f"CSV predictions saved to {args.output_csv}")


