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

def create_test_dataloader(csv_path, image_dir, batch_size=16):
    """
    Creates DataLoader for test dataset.
    """
    # Load Data
    df = pd.read_csv(csv_path)
    
    # Initialize Processor 
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
    
    # Create Dataset
    test_dataset = StreetViewDataset(df, image_dir, processor, is_test=True)
    
    # Create Loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    return test_loader, processor


class DirectionalImagesDataset(Dataset):
    """
    Dataset for test images organized by direction (north, east, south, west).
    Does not require a CSV file - images are loaded directly from the directory structure.
    """
    def __init__(self, image_dir, processor):
        self.image_dir = image_dir
        self.processor = processor
        
        # Get all unique image IDs by scanning the directory
        image_files = os.listdir(image_dir)
        image_ids = set()
        for img_file in image_files:
            # Extract image ID supporting both 'image_' and 'img_' prefixes
            parts = img_file.split('_')
            if len(parts) >= 3 and parts[0] in {"image", "img"}:
                image_id = parts[1]
                image_ids.add(image_id)
        
        self.image_ids = sorted(list(image_ids))
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        directions = ['north', 'east', 'south', 'west']
        images = []
        
        for direction in directions:
            # Try both filename patterns
            candidates = [
                os.path.join(self.image_dir, f"image_{image_id}_{direction}.jpg"),
                os.path.join(self.image_dir, f"img_{image_id}_{direction}.jpg"),
            ]
            image = None
            for img_path in candidates:
                image = cv2.imread(img_path)
                if image is not None:
                    break
            
            if image is None:
                # Fallback for missing/corrupt images
                image = np.zeros((256, 256, 3), dtype=np.uint8)
            else:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            images.append(image)
        
        # Process images using CLIP processor
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        pixel_values = inputs['pixel_values']
        
        return {
            'pixel_values': pixel_values,
            'image_id': image_id
        }


def create_directional_test_dataloader(test_image_dir, batch_size=16):
    """
    Creates DataLoader for test images organized by direction (north, east, south, west).
    
    Args:
        test_image_dir: Path to directory containing images named as "image_{id}_{direction}.jpg"
        batch_size: Batch size for DataLoader
    
    Returns:
        test_loader: DataLoader for test images
        processor: CLIPProcessor instance
    """
    # Initialize Processor
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
    
    # Create Dataset
    test_dataset = DirectionalImagesDataset(test_image_dir, processor)
    
    # Create Loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return test_loader, processor