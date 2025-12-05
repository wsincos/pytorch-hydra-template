"""
模型权重初始化工具
用于Transformer模型的标准初始化策略
"""
import math
import torch.nn as nn


def init_transformer_weights(module):
    """
    标准的Transformer权重初始化策略
    
    参考：
    1. "Attention is All You Need" 论文
    2. fairseq和Hugging Face的实现
    """
    if isinstance(module, nn.Linear):
        # Xavier uniform初始化（适合tanh/sigmoid激活）
        # 对于大多数Transformer的线性层都合适
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    
    elif isinstance(module, nn.Embedding):
        # 正态分布初始化，标准差 = 1/sqrt(embedding_dim)
        nn.init.normal_(module.weight, mean=0.0, std=module.embedding_dim ** -0.5)
        if module.padding_idx is not None:
            # padding token的embedding应该是0
            module.weight.data[module.padding_idx].zero_()
    
    elif isinstance(module, nn.LayerNorm):
        # LayerNorm: weight=1, bias=0
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    
    elif isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
        # 这些层内部的linear/layernorm会递归初始化
        pass


def init_output_projection(fc_layer, vocab_size):
    """
    特别处理输出投影层（decoder的最后一层）
    
    Args:
        fc_layer: 输出层的Linear模块
        vocab_size: 词表大小
    """
    # 权重使用Xavier初始化
    nn.init.xavier_uniform_(fc_layer.weight)
    
    # bias初始化为0，避免某些token天然具有更高的logits
    if fc_layer.bias is not None:
        nn.init.zeros_(fc_layer.bias)
        
        # 可选：为特殊token（如PAD）设置负bias
        # 这样可以降低模型生成PAD的概率
        # fc_layer.bias.data[0] = -10.0  # PAD token
