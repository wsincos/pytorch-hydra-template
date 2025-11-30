"""
Layers.py 存放一些'原子'的神经网络层组件, 例如各种位置编码模块等.
"""

import math
import torch
import torch.nn as nn

from src.utils.registry import LAYER_REGISTRY

@LAYER_REGISTRY.register()
class SinusoidalPE(nn.Module):
    """
    正弦位置编码 (Sinusoidal Positional Encoding)
    论文: Attention Is All You Need
    特点: 固定公式，不需要训练，能外推到比训练集更长的序列(理论上)。
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数维 sin, 奇数维 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer (不更新梯度，但随模型保存)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Batch, SeqLen, Dim]
        # 截取对应长度的位置编码叠加
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


@LAYER_REGISTRY.register()
class LearnablePE(nn.Module):
    """
    可学习位置编码 (Learnable Positional Encoding)
    论文: BERT / GPT 等模型常用
    特点: 位置信息是作为参数训练出来的，通常限制了最大序列长度(max_len)。
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 使用 Embedding 层来存储位置向量
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: [Batch, SeqLen, Dim]
        seq_len = x.size(1)
        
        # 生成位置索引: [0, 1, 2, ..., seq_len-1]
        # device=x.device 保证生成的 Tensor 和输入在同一个设备(GPU/CPU)上
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0) # [1, SeqLen]
        
        # 查表得到位置向量并叠加
        x = x + self.pe(positions)
        return self.dropout(x)