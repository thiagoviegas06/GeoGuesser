import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
from sklearn.model_selection import train_test_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class StreetViewDataset(Dataset):
    def __init__(self, df, image_dir, processor, is_test=False):
        self.df = df
        self.image_dir = image_dir
        self.processor = processor
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load all 4 directional images
        directions = ['north', 'east', 'south', 'west']
        images = []
        
        for direction in directions:
            img_name = row[f'image_{direction}']
            img_path = os.path.join(self.image_dir, img_name)
            
            # Use OpenCV to load image (Loads as BGR by default)
            image = cv2.imread(img_path)
            
            if image is None:
                # Fallback for missing/corrupt images
                # Create a black image: 256x256x3, uint8
                image = np.zeros((256, 256, 3), dtype=np.uint8)
            else:
                # Convert BGR (OpenCV standard) to RGB (CLIP requirement)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            images.append(image)

        # Process images using CLIP processor
        # CLIPProcessor accepts lists of NumPy arrays (H, W, C) directly
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        
        # The pixel_values will be shape [4, 3, 224, 224] (after processor resizing)
        pixel_values = inputs['pixel_values']

        if self.is_test:
            return {
                'pixel_values': pixel_values,
                'sample_id': row['sample_id']
            }
        else:
            # Targets
            state_idx = torch.tensor(row['state_idx'], dtype=torch.long)
            # GPS: [latitude, longitude]
            gps = torch.tensor([row['latitude'], row['longitude']], dtype=torch.float32)
            
            return {
                'pixel_values': pixel_values,
                'state_idx': state_idx,
                'gps': gps,
                'sample_id': row['sample_id']
            }

def create_dataloaders(csv_path, image_dir, batch_size=16):
    """
    Creates DataLoaders for training and validation.
    Splits the data 80/20.
    """
    # Load Data
    df = pd.read_csv(csv_path)
    
    # Split Data 80/20
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    
    # Initialize Processor 
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
    
    # Create Datasets
    train_dataset = StreetViewDataset(train_df, image_dir, processor)
    val_dataset = StreetViewDataset(val_df, image_dir, processor)
    
    # Create Loaders
    # num_workers can often be increased with OpenCV as it releases the GIL better than PIL in some versions
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, val_loader, processor