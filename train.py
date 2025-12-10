from config import BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS
from data_loader import create_dataloaders
from model import StreetCLIPMultiTask
import torch
import torch.nn as nn
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_score(state_logits, gps_pred, state_targets, gps_targets):
    """
    Implements the custom weighted scoring metric.
    Classification: Top-5 weighted (1.0, 0.6, 0.4, 0.25, 0.15)
    GPS: Usually GPS error is distance, but here we just need a score component.
         
         Using competition metric: 
         Score = 0.7 * Class_Score + 0.3 * GPS_Score
         (We will approximate GPS Score as 1.0 if dist < threshold, or just use 
          negative MSE for validation tracking since exact GPS scoring formula 
          wasn't provided, only the weight).
    """
    batch_size = state_targets.size(0)
    
    # --- Classification Score ---
    # Get top 5 predictions
    _, top5_preds = torch.topk(state_logits, 5, dim=1) # [Batch, 5]
    
    class_scores = torch.zeros(batch_size, device=state_logits.device)
    weights = torch.tensor([1.0, 0.60, 0.40, 0.25, 0.15], device=state_logits.device)
    
    for i in range(batch_size):
        target = state_targets[i]
        preds = top5_preds[i]
        # Find first match
        match_idx = (preds == target).nonzero(as_tuple=True)[0]
        if len(match_idx) > 0:
            # Take the first match only
            rank = match_idx[0].item()
            class_scores[i] = weights[rank]
            
    avg_class_score = class_scores.mean().item()
    
    # --- GPS Error (MSE) ---
    # We track MSE for monitoring, although not the competition score format
    mse = nn.functional.mse_loss(gps_pred, gps_targets).item()
    
    return avg_class_score, mse

def train_and_validate(args):
    # Setup Data
    train_loader, val_loader, _ = create_dataloaders(args.csv_path, args.image_dir, BATCH_SIZE)
    
    # Setup Model
    model = StreetCLIPMultiTask(num_states=50).to(DEVICE)
    
    checkpoint_loaded = False
    if args.checkpoint:
        print(f"Resuming training from checkpoint: {args.checkpoint}")
        try:
            # Load the state dictionary
            checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
            
            # Handle cases where the checkpoint might be the full model object or just state_dict
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # Common pattern if saving optimizer state + model state
                model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict):
                # Standard state_dict save
                model.load_state_dict(checkpoint)
            else:
                # If entire model was saved (not recommended but possible)
                model.load_state_dict(checkpoint.state_dict())
                
            print("Checkpoint loaded successfully.")
            checkpoint_loaded = True

        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return
    else:
        print("No checkpoint provided. Starting fresh training.")

    # Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion_class = nn.CrossEntropyLoss()
    criterion_gps = nn.MSELoss() 
    
    best_val_score = -float('inf')

    if checkpoint_loaded:
        print("Running initial validation to set baseline score...")
        val_c, val_g = validate(model, val_loader, DEVICE, desc="Initial Val")
        # Metric: Maximize Class Score, Minimize GPS Error
        best_val_score = val_c - (val_g * 0.001)
        print(f"Baseline loaded: Class Score: {val_c:.4f}, GPS MSE: {val_g:.4f}, Best Score: {best_val_score:.4f}")
        print("-" * 30)
    
    for epoch in range(NUM_EPOCHS):
        # --- TRAINING LOOP ---
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for batch in loop:
            pixel_values = batch['pixel_values'].to(DEVICE)
            state_targets = batch['state_idx'].to(DEVICE)
            gps_targets = batch['gps'].to(DEVICE)
            
            # Forward
            state_logits, gps_pred = model(pixel_values)
            
            # Loss Calculation
            loss_cls = criterion_class(state_logits, state_targets)
            loss_gps = criterion_gps(gps_pred, gps_targets)
            
            # Combine losses (Weighted sum is often necessary to balance gradients)
            # GPS loss can be large, so we might scale it down or up. 
            # A common heuristic is ensuring both losses are roughly same magnitude.
            loss = loss_cls + (loss_gps * 0.1) 
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION LOOP ---
        model.eval()
        val_class_score = 0
        val_gps_mse = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                pixel_values = batch['pixel_values'].to(DEVICE)
                state_targets = batch['state_idx'].to(DEVICE)
                gps_targets = batch['gps'].to(DEVICE)
                
                state_logits, gps_pred = model(pixel_values)
                
                # Calculate metric scores
                c_score, g_mse = calculate_score(state_logits, gps_pred, state_targets, gps_targets)
                val_class_score += c_score
                val_gps_mse += g_mse
        
        avg_val_class_score = val_class_score / len(val_loader)
        avg_val_gps_mse = val_gps_mse / len(val_loader)
        
        # Construct a composite metric for "Best Model" selection
        # Since we want High Class Score and Low GPS Error:
        # We can treat GPS MSE as a penalty.
        current_score = avg_val_class_score - (avg_val_gps_mse * 0.001) 
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Class Score (Weighted Top-5): {avg_val_class_score:.4f} / 1.0")
        print(f"Val GPS MSE: {avg_val_gps_mse:.4f}")
        
        # Save Best Model
        if current_score > best_val_score:
            best_val_score = current_score
            torch.save(model.state_dict(), "best_streetclip_model.pth")
            print(">>> New Best Model Saved!")
        print("-" * 30)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    if DEVICE != "cuda":
        raise ImportError("Warning: CUDA device not available. Training will be slow on CPU.")
    
    parser = argparse.ArgumentParser(description="Train StreetCLIP for Geolocalization")
    # Required arguments
    parser.add_argument('--csv_path', type=str, default='kaggle_dataset/train_ground_truth.csv', help='Path to ground truth CSV')
    parser.add_argument('--image_dir', type=str, default='kaggle_dataset/train_images/', help='Path to image directory')
    
    # Optional checkpoint flag
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to .pth file to resume training')
    
    args = parser.parse_args()
    
    train_and_validate(args)

