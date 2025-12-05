import math
import torch.nn as nn
from src.utils.registry import ENCODER_REGISTRY, LAYER_REGISTRY

@ENCODER_REGISTRY.register("TransformerEncoder")
class MyEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, pe_cfg, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoder = LAYER_REGISTRY.get(pe_cfg.pop("name"))(**pe_cfg)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.model = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # 1. Embedding
        x = self.embedding(x)
        
        # 2. Scaling
        x = x * math.sqrt(self.d_model)
        
        # 3. Add Positional Encoding
        x = self.pos_encoder(x)
        
        return self.model(x)