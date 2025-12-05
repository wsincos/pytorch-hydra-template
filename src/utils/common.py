"""
src/utils/common.py 通常存放那些没有任何业务逻辑依赖、整个项目通用的、最底层的辅助函数。

一般包括如下内容：

1. 实验可复现性：随机种子管理 (Seed Management)
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # 甚至可能包含更严格的 CuDNN 确定性设置
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

2. 第三方组件的注册 (Registration)
    def register_standard_components():
        from .registry import OPTIMIZER_REGISTRY, CRITERION_REGISTRY
        
        # 注册优化器
        OPTIMIZER_REGISTRY.register("Adam")(torch.optim.Adam)
        OPTIMIZER_REGISTRY.register("SGD")(torch.optim.SGD)
        
        # 注册 Loss
        CRITERION_REGISTRY.register("CrossEntropy")(torch.nn.CrossEntropyLoss)


3. 模型统计与工具 (Model Utilities), 用来统计参数量、打印模型结构，或者处理梯度裁剪等通用操作。
    def count_parameters(model): # 统计模型有多少可训练参数
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def grad_norm(model): # 计算梯度的范数，用于监控训练稳定性
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5

4. 硬件与环境抽象 (Device & Distributed)
    def get_device(cfg):
        if cfg.device == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif cfg.device == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps') # Mac 专用
        return torch.device('cpu')

"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import omegaconf
from .registry import OPTIMIZER_REGISTRY, CRITERION_REGISTRY, SCHEDULER_REGISTRY


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def dict_from_config(cfg: omegaconf.DictConfig) -> dict:
    """将配置对象转换为普通字典，方便传递参数"""
    # to_container: 将 OmegaConf 对象转换为原生 Python 对象
    # resolve=True: 解析所有的引用和插值，确保返回的字典中不包含任何 OmegaConf 特有的结构
    return omegaconf.OmegaConf.to_container(cfg, resolve=True)

def register_standard_components():
    # optimizers
    OPTIMIZER_REGISTRY.register("Adam")(optim.Adam)
    OPTIMIZER_REGISTRY.register("SGD")(optim.SGD)
    OPTIMIZER_REGISTRY.register("AdamW")(optim.AdamW)

    # criterions
    CRITERION_REGISTRY.register("CrossEntropyLoss")(nn.CrossEntropyLoss)
    CRITERION_REGISTRY.register("MSELoss")(nn.MSELoss)
    CRITERION_REGISTRY.register("L1Loss")(nn.L1Loss)

    # schedulers
    SCHEDULER_REGISTRY.register("StepLR")(optim.lr_scheduler.StepLR) # 学习率调度器
    SCHEDULER_REGISTRY.register("ReduceLROnPlateau")(optim.lr_scheduler.ReduceLROnPlateau) # 动态调整学习率
    SCHEDULER_REGISTRY.register("CosineAnnealingLR")(optim.lr_scheduler.CosineAnnealingLR) # 余弦退火的方式调整学习率
    SCHEDULER_REGISTRY.register("OneCycleLR")(optim.lr_scheduler.OneCycleLR) # One Cycle 学习率调度器