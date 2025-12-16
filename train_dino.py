import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from data_loader_dino import create_dino_dataloaders
from dinomodel import StreetDinoV2MultiTask

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def calculate_score(state_logits, gps_pred, state_targets, gps_targets):
    """Custom weighted score for validation."""
    batch_size = state_targets.size(0)

    # Classification Top-5 weighted
    _, top5_preds = torch.topk(state_logits, 5, dim=1)  # [Batch, 5]
    class_scores = torch.zeros(batch_size, device=state_logits.device)
    weights = torch.tensor([1.0, 0.60, 0.40, 0.25, 0.15], device=state_logits.device)

    for i in range(batch_size):
        target = state_targets[i]
        preds = top5_preds[i]
        match_idx = (preds == target).nonzero(as_tuple=True)[0]
        if len(match_idx) > 0:
            rank = match_idx[0].item()
            class_scores[i] = weights[rank]

    avg_class_score = class_scores.mean().item()

    # GPS MSE
    mse = nn.functional.mse_loss(gps_pred, gps_targets).item()

    return avg_class_score, mse


def validate(model, val_loader, device, desc="Validation"):
    model.eval()
    val_class_score = 0
    val_gps_mse = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=desc):
            pixel_values = batch['pixel_values'].to(device)
            state_targets = batch['state_idx'].to(device)
            gps_targets = batch['gps'].to(device)

            state_logits, gps_pred = model(pixel_values)
            c_score, g_mse = calculate_score(state_logits, gps_pred, state_targets, gps_targets)
            val_class_score += c_score
            val_gps_mse += g_mse

    avg_val_class_score = val_class_score / len(val_loader)
    avg_val_gps_mse = val_gps_mse / len(val_loader)

    return avg_val_class_score, avg_val_gps_mse


def train_and_validate(args, batch_size=4):
    # Data
    train_loader, val_loader, _ = create_dino_dataloaders(
        args.csv_path, args.image_dir, batch_size=batch_size, model_name=args.model_name
    )

    # Model
    model = StreetDinoV2MultiTask(num_states=50, model_name=args.model_name).to(DEVICE)

    checkpoint_loaded = False
    if args.checkpoint:
        print(f"Resuming training from checkpoint: {args.checkpoint}")
        try:
            checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint.state_dict())
            print("Checkpoint loaded successfully.")
            checkpoint_loaded = True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return
    else:
        print("No checkpoint provided. Starting fresh training.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion_class = nn.CrossEntropyLoss()
    criterion_gps = nn.MSELoss()

    best_val_score = -float('inf')

    if checkpoint_loaded:
        print("Running initial validation to set baseline score...")
        val_c, val_g = validate(model, val_loader, DEVICE, desc="Initial Val")
        best_val_score = val_c - (val_g * 0.001)
        print(f"Baseline loaded: Class Score: {val_c:.4f}, GPS MSE: {val_g:.4f}, Best Score: {best_val_score:.4f}")
        print("-" * 30)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for batch in loop:
            pixel_values = batch['pixel_values'].to(DEVICE)
            state_targets = batch['state_idx'].to(DEVICE)
            gps_targets = batch['gps'].to(DEVICE)

            state_logits, gps_pred = model(pixel_values)

            loss_cls = criterion_class(state_logits, state_targets)
            loss_gps = criterion_gps(gps_pred, gps_targets)
            loss = loss_cls + (loss_gps * 0.1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        avg_val_class_score, avg_val_gps_mse = validate(
            model, val_loader, DEVICE, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"
        )

        current_score = avg_val_class_score - (avg_val_gps_mse * 0.001)

        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Class Score (Weighted Top-5): {avg_val_class_score:.4f} / 1.0")
        print(f"Val GPS MSE: {avg_val_gps_mse:.4f}")

        if current_score > best_val_score:
            best_val_score = current_score
            torch.save(model.state_dict(), "best_streetdino_model.pth")
            print(">>> New Best Model Saved!")
        print("-" * 30)


def main():
    parser = argparse.ArgumentParser(description="Train StreetDINO for Geolocalization")
    parser.add_argument('--csv_path', type=str, default='kaggle_dataset/train_ground_truth.csv', help='Path to ground truth CSV')
    parser.add_argument('--image_dir', type=str, default='kaggle_dataset/train_images/', help='Path to image directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to .pth file to resume training')
    parser.add_argument('--model_name', type=str, default='facebook/dinov2-base', help='DINOv2 model name')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training (default 4 for CPU, use 16-32 for GPU)')
    args = parser.parse_args()

    train_and_validate(args, batch_size=args.batch_size)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    if DEVICE != "cuda":
        print("Warning: CUDA device not available. Training will be slow on CPU.")
    main()
