import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor
from sklearn.model_selection import train_test_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class StreetViewDinoDataset(Dataset):
    def __init__(self, df, image_dir, processor, is_test=False):
        self.df = df
        self.image_dir = image_dir
        self.processor = processor
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        directions = ['north', 'east', 'south', 'west']
        images = []

        for direction in directions:
            img_name = row[f'image_{direction}']
            img_path = os.path.join(self.image_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                image = np.zeros((256, 256, 3), dtype=np.uint8)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)

        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values']

        if self.is_test:
            return {
                'pixel_values': pixel_values,
                'sample_id': row['sample_id']
            }
        else:
            state_idx = torch.tensor(row['state_idx'], dtype=torch.long)
            gps = torch.tensor([row['latitude'], row['longitude']], dtype=torch.float32)
            return {
                'pixel_values': pixel_values,
                'state_idx': state_idx,
                'gps': gps,
                'sample_id': row['sample_id']
            }


def create_dino_dataloaders(csv_path, image_dir, batch_size=16, model_name="facebook/dinov2-base"):
    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    processor = AutoImageProcessor.from_pretrained(model_name)

    train_dataset = StreetViewDinoDataset(train_df, image_dir, processor)
    val_dataset = StreetViewDinoDataset(val_df, image_dir, processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, processor


def create_dino_test_dataloader(csv_path, image_dir, batch_size=16, model_name="facebook/dinov2-base"):
    df = pd.read_csv(csv_path)
    processor = AutoImageProcessor.from_pretrained(model_name)

    test_dataset = StreetViewDinoDataset(df, image_dir, processor, is_test=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return test_loader, processor
