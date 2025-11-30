import math
import torch.nn as nn
from src.utils.registry import DECODER_REGISTRY, LAYER_REGISTRY

@DECODER_REGISTRY.register("TransformerDecoder")
class MyDecoder(nn.Module):
    def __init__(self, output_dim, d_model, nhead, num_layers, pe_cfg, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(output_dim, d_model)
        
        self.pos_decoder = LAYER_REGISTRY.get(pe_cfg.pop("name"))(**pe_cfg)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.model = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        
    def forward(self, tgt, memory):
        # [修改] 1. Embedding
        tgt = self.embedding(tgt)
        
        # [修改] 2. Scaling
        tgt = tgt * math.sqrt(self.d_model)
        
        # [修改] 3. Add Position Encoding
        tgt = self.pos_decoder(tgt)
        
        out = self.model(tgt, memory)
        return self.fc(out)