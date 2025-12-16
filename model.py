import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPVisionModel

class StreetCLIPMultiTask(nn.Module):
    def __init__(self, num_states=50, fusion_alpha=0.5):
        super(StreetCLIPMultiTask, self).__init__()
        self.backbone = CLIPVisionModel.from_pretrained("geolocal/StreetCLIP")
        hidden_size = self.backbone.config.hidden_size 
        self.num_states = num_states
        # Fusion alpha balances max-prob vs inverse-entropy confidence [0..1]
        self.fusion_alpha = fusion_alpha

        # State Classification
        self.state_classifier = nn.Linear(hidden_size, num_states)
        
        # GPS Regression
        self.gps_regressor = nn.Linear(hidden_size, 2)
        
        # Define US Geographic Bounds (Lat, Lon) to clip guesses outide of bounds.
        # Lat: ~18 (Hawaii South) to ~72 (Alaska North)
        # Lon: ~-180 (Alaska West) to ~-65 (Maine East)
        self.register_buffer('lat_range', torch.tensor([18.0, 72.0]))
        self.register_buffer('lon_range', torch.tensor([-180.0, -65.0]))

    def forward(self, pixel_values):
        b, n, c, h, w = pixel_values.shape
        stacked_images = pixel_values.view(b * n, c, h, w)
        
        # Get Features
        outputs = self.backbone(pixel_values=stacked_images)
        embeddings = outputs.pooler_output                        # [B*N, H]
        embeddings = embeddings.view(b, n, -1)                    # [B, N, H]
        
        # Direction-wise confidence from per-direction state logits
        # Compute provisional logits per direction
        dir_logits = self.state_classifier(embeddings)            # [B, N, num_states]
        dir_probs = torch.softmax(dir_logits, dim=-1)             # [B, N, num_states]

        # Confidence 1: Max softmax probability per direction
        conf_maxprob = dir_probs.max(dim=-1).values               # [B, N]

        # Confidence 2: Inverse normalized entropy per direction
        # Entropy H(p) = -sum p*log(p), normalize by log(K)
        eps = 1e-8
        entropy = -(dir_probs * (dir_probs + eps).log()).sum(dim=-1)             # [B, N]
        entropy_norm = entropy / (torch.log(torch.tensor(float(self.num_states), device=entropy.device)))
        conf_inv_entropy = 1.0 - entropy_norm                                      # [B, N] higher = more confident

        # Combine confidences with fusion_alpha
        dir_conf = self.fusion_alpha * conf_maxprob + (1.0 - self.fusion_alpha) * conf_inv_entropy  # [B, N]

        # Normalize weights to sum to 1 per sample
        weights = dir_conf / (dir_conf.sum(dim=1, keepdim=True) + 1e-8)  # [B, N]
        
        # Weighted fuse of embeddings across directions
        fused_embedding = (embeddings * weights.unsqueeze(-1)).sum(dim=1) # [B, H]
        
        # State Prediction (Unchanged)
        state_logits = self.state_classifier(fused_embedding)
        
        # GPS Prediction (Constrained)
        # Raw linear output: (-inf, +inf)
        raw_gps = self.gps_regressor(fused_embedding)
        
        # Apply Sigmoid to map to (0, 1)
        norm_gps = torch.sigmoid(raw_gps)
        
        # Scale to actual Lat/Lon ranges
        # lat = 0..1 * (Max - Min) + Min
        # lon = 0..1 * (Max - Min) + Min
        
        lat_pred = norm_gps[:, 0] * (self.lat_range[1] - self.lat_range[0]) + self.lat_range[0]
        lon_pred = norm_gps[:, 1] * (self.lon_range[1] - self.lon_range[0]) + self.lon_range[0]
        
        # Stack back into [Batch, 2]
        gps_pred = torch.stack([lat_pred, lon_pred], dim=1)
        
        return state_logits, gps_pred