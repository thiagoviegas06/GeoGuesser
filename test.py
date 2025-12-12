import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor

# Import your custom classes
from data_loader import StreetViewDataset
from data_loader import create_dataloaders
from model import StreetCLIPMultiTask

# Configuration
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_predictions(args):
    print(f"Using device: {DEVICE}")

    # 1. Load the Sample Submission CSV
    # We use this as the "index" for what we need to predict
    print(f"Reading submission template from {args.submission_csv}...")
    df = pd.read_csv(args.submission_csv)

    # 2. Setup Data
    print("Initializing Test DataLoader...")
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

    # Initialize Dataset with is_test=True
    # This ensures __getitem__ returns only pixel_values and sample_id
    test_dataset = StreetViewDataset(
        df=df,
        image_dir=args.image_dir,
        processor=processor,
        is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # Important: Don't shuffle for testing!
        num_workers=4,
        pin_memory=True
    )

    # 3. Setup Model & Load Checkpoint
    print(f"Loading model from {args.model_path}...")
    model = StreetCLIPMultiTask(num_states=50).to(DEVICE)

    try:
        checkpoint = torch.load(args.model_path, map_location=DEVICE)
        # Handle state_dict vs full checkpoint dict
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # 4. Inference Loop
    print("Starting Inference...")

    # Dictionaries to store results: sample_id -> data
    results = {
        'state_preds': {}, # sample_id: [top1, top2, top3, top4, top5]
        'gps_preds': {}    # sample_id: (lat, lon)
    }

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            pixel_values = batch['pixel_values'].to(DEVICE)
            sample_ids = batch['sample_id'] # List or Tensor of IDs

            # Forward Pass
            state_logits, gps_pred = model(pixel_values)

            # --- Process State Predictions ---
            # Get Top 5 indices
            _, top5_indices = torch.topk(state_logits, 5, dim=1)

            # --- Process Results for this batch ---
            top5_indices = top5_indices.cpu().numpy()
            gps_pred = gps_pred.cpu().numpy()

            for i, s_id in enumerate(sample_ids):
                # Ensure s_id is a standard python integer/string, not tensor
                if isinstance(s_id, torch.Tensor):
                    s_id = s_id.item()

                results['state_preds'][s_id] = top5_indices[i]
                results['gps_preds'][s_id] = gps_pred[i]

    # 5. Fill DataFrame
    print("Formatting submission file...")

    # We iterate through the DataFrame to ensure we fill the correct rows
    for index, row in df.iterrows():
        s_id = row['sample_id']

        if s_id in results['state_preds']:
            # Fill States
            preds = results['state_preds'][s_id]
            df.at[index, 'predicted_state_idx_1'] = preds[0]
            df.at[index, 'predicted_state_idx_2'] = preds[1]
            df.at[index, 'predicted_state_idx_3'] = preds[2]
            df.at[index, 'predicted_state_idx_4'] = preds[3]
            df.at[index, 'predicted_state_idx_5'] = preds[4]

            # Fill GPS
            lat, lon = results['gps_preds'][s_id]
            df.at[index, 'predicted_latitude'] = lat
            df.at[index, 'predicted_longitude'] = lon

    # 6. Save
    output_filename = "submission.csv"
    df.to_csv(output_filename, index=False)
    print(f"Done! Submission saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Predictions for StreetCLIP")

    parser.add_argument('--submission_csv', type=str,
                        default='kaggle_dataset/sample_submission.csv',
                        help='Path to sample_submission.csv')

    parser.add_argument('--image_dir', type=str,
                        default='kaggle_dataset/test_images',
                        help='Path to test images directory')

    parser.add_argument('--model_path', type=str,
                        default='best_streetclip_model.pth',
                        help='Path to saved .pth model checkpoint')

    args = parser.parse_args()

    generate_predictions(args)
