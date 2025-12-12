import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class StreetDinoV2MultiTask(nn.Module):
    """
    Multi-task model using DINOv2 backbone for geolocation prediction.
    Predicts US state (classification) and GPS coordinates (regression).
    """
    def __init__(self, num_states=50, model_name="facebook/dinov2-base"):
        super(StreetDinoV2MultiTask, self).__init__()
        # Load DINOv2 backbone
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # DINOv2 base has 768 hidden size, large has 1024
        # Infer from config
        hidden_size = self.backbone.config.hidden_size
        
        # State Classification
        self.state_classifier = nn.Linear(hidden_size, num_states)
        
        # GPS Regression
        self.gps_regressor = nn.Linear(hidden_size, 2)
        
        # Define US Geographic Bounds (Lat, Lon)
        # Lat: ~18 (Hawaii South) to ~72 (Alaska North)
        # Lon: ~-180 (Alaska West) to ~-65 (Maine East)
        self.register_buffer('lat_range', torch.tensor([18.0, 72.0]))
        self.register_buffer('lon_range', torch.tensor([-180.0, -65.0]))
    
    def forward(self, pixel_values):
        """
        Forward pass for multi-directional street view images.
        
        Args:
            pixel_values: [Batch, N_directions, C, H, W] tensor
        
        Returns:
            state_logits: [Batch, num_states] logits for state classification
            gps_pred: [Batch, 2] predicted [latitude, longitude]
        """
        # pixel_values: [Batch, N_directions, C, H, W]
        b, n, c, h, w = pixel_values.shape
        stacked_images = pixel_values.view(b * n, c, h, w)
        
        # Get Features from DINOv2
        outputs = self.backbone(pixel_values=stacked_images)
        
        # DINOv2 uses CLS token from last_hidden_state
        # Shape: [B*N, seq_len, hidden_size]
        # CLS token is the first token
        embeddings = outputs.last_hidden_state[:, 0, :]  # [B*N, hidden_size]
        
        # Reshape and fuse across directions
        embeddings = embeddings.view(b, n, -1)  # [B, N, hidden_size]
        fused_embedding = torch.mean(embeddings, dim=1)  # [B, hidden_size]
        
        # State Prediction
        state_logits = self.state_classifier(fused_embedding)
        
        # GPS Prediction (Constrained)
        raw_gps = self.gps_regressor(fused_embedding)
        
        # Apply Sigmoid to map to (0, 1)
        norm_gps = torch.sigmoid(raw_gps)
        
        # Scale to actual Lat/Lon ranges
        lat_pred = norm_gps[:, 0] * (self.lat_range[1] - self.lat_range[0]) + self.lat_range[0]
        lon_pred = norm_gps[:, 1] * (self.lon_range[1] - self.lon_range[0]) + self.lon_range[0]
        
        # Stack back into [Batch, 2]
        gps_pred = torch.stack([lat_pred, lon_pred], dim=1)
        
        return state_logits, gps_pred
