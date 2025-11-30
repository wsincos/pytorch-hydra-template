"""
读取 Config 并实例化Transformer模型组件的模块。
"""
from curses import wrapper
import logging
import omegaconf
from src.utils.registry import (ENCODER_REGISTRY, DECODER_REGISTRY, MODEL_REGISTRY)
from src.utils.common import dict_from_config

logger = logging.getLogger(__name__)

def get_model(cfg: omegaconf.DictConfig):
    """
    根据配置实例化Transformer模型组件。

    参数:
        cfg (omegaconf.DictConfig): 包含模型组件配置的字典。

    返回:
        model: 实例化的Transformer模型。
    """
    model_name = cfg.model.name
    
    if model_name == 'seq2seq_transformer':
        # 1. 提取框架配置(所有组件的配置都在其中)
        arch_cfg = dict_from_config(cfg.model.arch)
        
        # 2. 分离配置
        encoder_cfg = arch_cfg.pop("encoder")
        decoder_cfg = arch_cfg.pop("decoder")

        # 3. 实例化组件
        encoder_name = encoder_cfg.pop("name")
        logger.info(f"[Builder] Instantiating Encoder: {encoder_name}")
        encoder = ENCODER_REGISTRY.get(encoder_name)(**encoder_cfg)

        decoder_name = decoder_cfg.pop("name")
        logger.info(f"[Builder] Instantiating Decoder: {decoder_name}")
        decoder = DECODER_REGISTRY.get(decoder_name)(**decoder_cfg)

        # 4. 实例化模型
        wrapper_cfg = dict_from_config(cfg.model.wrapper)
        wrapper_name = wrapper_cfg.pop("name")
        logger.info(f"[Builder] Instantiating Model: {wrapper_name}")
        wrapper_cls = MODEL_REGISTRY.get(wrapper_name)
        return wrapper_cls(encoder=encoder, decoder=decoder, **wrapper_cfg)
    else:
        raise KeyError(f"Unknown model name: {model_name}")