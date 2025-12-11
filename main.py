import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
#import torch # Added import for torch
#from torchvision import models, transforms
from dataset_loader import load_dataset
from train import extract_ground_truth


if __name__ == "__main__":
    training_path = "./kaggle_dataset/train_images"
    images_dict = load_dataset(training_path)
    print(f"Loaded {len(images_dict)} categories.")