import math
import torch.nn as nn
from src.utils.registry import ENCODER_REGISTRY, LAYER_REGISTRY
from .layers import xavier_init_weights

@ENCODER_REGISTRY.register("TransformerEncoder")
class MyEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, pe_cfg, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.dim_feedforward = dim_feedforward
        self.embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoder = LAYER_REGISTRY.get(pe_cfg.pop("name"))(**pe_cfg)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=self.dim_feedforward,
            dropout=dropout, 
            batch_first=True
        )
        self.model = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        # 初始化权重
        self.apply(xavier_init_weights)
        
    def forward(self, x):
        # 1. Embedding
        x = self.embedding(x)
        
        # 2. Scaling
        x = x * math.sqrt(self.d_model)
        
        # 3. Add Positional Encoding
        x = self.pos_encoder(x)
        
        return self.model(x)
