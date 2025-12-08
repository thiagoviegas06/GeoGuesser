import os
import torch

# Configuration
BATCH_SIZE = 16 # Adjust based on your GPU VRAM. Multi-view takes 4x memory per sample.
LEARNING_RATE = 1e-5
NUM_EPOCHS = 20
IMAGE_SIZE = 224 # CLIP standard input size