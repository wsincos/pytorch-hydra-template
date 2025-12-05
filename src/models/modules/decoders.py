import math
import torch
import torch.nn as nn
from src.utils.registry import DECODER_REGISTRY, LAYER_REGISTRY
from .layers import xavier_init_weights

@DECODER_REGISTRY.register("TransformerDecoder")
class MyDecoder(nn.Module):
    def __init__(self, output_dim, d_model, nhead, num_layers, dim_feedforward, pe_cfg, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        self.dim_feedforward = dim_feedforward
        self.embedding = nn.Embedding(output_dim, d_model)
        
        self.pos_decoder = LAYER_REGISTRY.get(pe_cfg.pop("name"))(**pe_cfg)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward = self.dim_feedforward,
            dropout=dropout, 
            batch_first=True
        )
        self.model = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)
        
        # 初始化权重
        self.apply(xavier_init_weights)
        
        
    def forward(self, tgt, memory):
        # 1. Embedding
        tgt = self.embedding(tgt)
        
        # 2. Scaling
        tgt = tgt * math.sqrt(self.d_model)
        
        # 3. Add Position Encoding
        tgt = self.pos_decoder(tgt)
        
        # 生成 causal mask，防止 decoder 在训练阶段看到未来的 target token
        seq_len = tgt.size(1)
        # PyTorch Transformer 使用 bool mask: True = 被mask的位置
        tgt_mask = torch.triu(torch.ones((seq_len, seq_len), device=tgt.device, dtype=torch.bool), diagonal=1)

        out = self.model(tgt, memory, tgt_mask=tgt_mask)
        return self.fc(out)
