"""
Solver: 深度学习实验的核心控制器 (The Core Controller)

设计理念 (Design Philosophy):
    Solver 类是连接 "静态配置 (Configuration)" 与 "动态执行 (Execution)" 的桥梁。
    它遵循 Meta/Fairseq 的工程范式，负责管理整个训练生命周期。

核心职责 (Core Responsibilities):
    1. 组装 (Assembly): 
       根据 Hydra 配置 (`cfg`)，通过 Registry 动态实例化模型、优化器、数据集和损失函数。
       它解决了 "组件之间如何连接" 的问题。
       
    2. 调度 (Orchestration):
       控制训练的主循环 (Epoch Loop)，负责数据加载、前向传播、反向传播和参数更新。
       它将 `run_step` 定义为原子操作，保证了训练逻辑的清晰。

    3. 状态管理 (State Management):
        负责 Checkpoint 的保存与加载 (Resume)。它维护着模型的权重、优化器的动量以及当前的训练步数。

    4. 观测与反馈 (Observability):
       集成 WandB 或 TensorBoard, 实时记录 Loss 和评估指标，负责在训练过程中执行 Evaluate 逻辑。

使用方式:
    solver = Solver(cfg)
    solver.train()
"""
import os
import logging
import torch
import wandb
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig

from src.models.builders import get_model
from src.utils.common import register_standard_components, seed_everything
from src.data.datasets import get_dataloader
from src.utils.registry import OPTIMIZER_REGISTRY, CRITERION_REGISTRY, SCHEDULER_REGISTRY

logger = logging.getLogger(__name__)

class Solver:
    def __init__(self, cfg):
        self.cfg = cfg
        seed_everything(cfg.train.seed)
        register_standard_components()
        self.device = torch.device(cfg.train.device)

        # 1. 构建数据
        logger.info("Building Train Loader...")
        self.train_loader = get_dataloader(cfg, split='train')
        
        logger.info("Building Test Loader...")
        self.test_loader = get_dataloader(cfg, split='test')

        # 2. 自动调整模型词表大小 (适配 BERT)
        if hasattr(self.train_loader.dataset, 'get_vocab_size'):
            vocab_size = self.train_loader.dataset.get_vocab_size()
            logger.info(f"Auto-setting vocab size from Tokenizer: {vocab_size}")
            
            cfg.model.shared.src_vocab_size = vocab_size
            cfg.model.shared.tgt_vocab_size = vocab_size
        
        # 3. 构建模型
        self.model = get_model(cfg).to(self.device)

        # 4. 构建优化器
        optim_cls = OPTIMIZER_REGISTRY.get(cfg.optimizer.name)
        self.optimizer = optim_cls(self.model.parameters(), **cfg.optimizer.params)

        # 5. 构建 Loss (关键：忽略 Padding)
        loss_cls = CRITERION_REGISTRY.get(cfg.criterion.name)
        # BERT 的 pad_token_id 通常是 0
        self.criterion = loss_cls(ignore_index=0, **cfg.criterion.params)

        # 6. 构建 Scheduler
        scheduler_cls = SCHEDULER_REGISTRY.get(cfg.scheduler.name)
        self.scheduler = scheduler_cls(self.optimizer, **cfg.scheduler.params)

        # 7. 初始化 Checkpoint 目录
        if cfg.checkpoint.enabled:
            # 1. 获取 Hydra 自动生成的本次实验目录 (outputs/日期/时间)
            try:
                run_dir = HydraConfig.get().runtime.output_dir
            except Exception:
                # 兜底：如果不是通过 hydra 启动（比如 debug），就存到当前目录
                run_dir = "."
            
            # 2. 拼接路径： outputs/.../checkpoints
            self.save_dir = os.path.join(run_dir, cfg.checkpoint.save_dir)
            
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"[Solver] Checkpoints will be saved to: {self.save_dir}")
        self.best_loss = float('inf')

        # 8. WandB
        if cfg.logger.enable:
            # 将配置转为字典
            logger_cfg = OmegaConf.to_container(cfg.logger, resolve=True)
            # 剔除 'enable' 字段，因为它是给 if 判断用的，不是给 wandb.init 用的
            logger_cfg.pop('enable')
            
            wandb.init(
                # 使用 **解包，这样 name, tags, group, mode 等所有参数都会自动传进去
                **logger_cfg, 
                config=OmegaConf.to_container(cfg, resolve=True)
            )
    
    def run_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        
        tgt_input = y[:, :-1] # 输入: 去掉尾部
        tgt_label = y[:, 1:]  # 标签: 去掉头部 (预测下一个词)
        
        logits = self.model(x, tgt_input)
        
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_label.reshape(-1))
        
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪 (防止真实数据导致梯度爆炸)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()
    

    def save_checkpoint(self, epoch, is_best=False):
        if not self.cfg.checkpoint.enabled:
            return

        # 构造要保存的字典（Meta 风格的标准包）
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),      # 模型参数
            "optimizer_state": self.optimizer.state_dict(), # 优化器状态 (恢复训练必须!)
            "best_loss": self.best_loss,
            "config": OmegaConf.to_container(self.cfg, resolve=True) # 保存当前配置，防止以后忘了参数
        }
        
        # 1. 始终覆盖保存 last.pt (用于恢复)
        last_path = os.path.join(self.save_dir, "checkpoint_last.pt")
        torch.save(state, last_path)

        # 记录 Epoch 存档 (可选)
        # if (epoch + 1) % self.cfg.checkpoint.save_every == 0:
        #     epoch_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        #     torch.save(state, epoch_path)
        
        # 2. 如果是最佳模型，额外保存 best.pt (用于推理)
        if is_best:
            best_path = os.path.join(self.save_dir, "checkpoint_best.pt")
            torch.save(state, best_path)
            logger.info(f"New best checkpoint saved: {best_path}")
        else:
            logger.info(f"Checkpoint saved: {last_path}")
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x, y = x.to(self.device), y.to(self.device)
                
                tgt_input = y[:, :-1]
                tgt_label = y[:, 1:]
                
                logits = self.model(x, tgt_input)
                
                loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_label.reshape(-1))
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.test_loader)
        logger.info(f"=== Evaluation Done | Avg Loss: {avg_loss:.4f} ===")
        
        self.model.train()
        return avg_loss


    def train(self):
        logger.info(f"Start Training Process... Total Epochs: {self.cfg.train.epochs}")
        self.model.train()
        
        for epoch in range(self.cfg.train.epochs):
            # --- 1. 训练主循环 (Training Loop) ---
            total_loss = 0
            for i, (x, y) in enumerate(self.train_loader):
                loss = self.run_step(x, y)
                total_loss += loss
                
                if i % 50 == 0:
                    logger.info(f"Epoch {epoch} | Step {i} | Train Loss: {loss:.4f}")
                    if self.cfg.logger.enable:
                        wandb.log({"train/step_loss": loss})
            
            # 计算并打印 Epoch 平均 Loss
            avg_train_loss = total_loss / len(self.train_loader)
            logger.info(f"=== Epoch {epoch} Summary | Avg Train Loss: {avg_train_loss:.4f} ===")
            if self.cfg.logger.enable:
                wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch})
            

            # --- 2. 评估并保存模型 ---
            test_loss = self.evaluate()

            self.is_best = False
            if test_loss < self.best_loss:
                logger.info(f"New Best Model Found! (Loss: {self.best_loss:.4f} -> {test_loss:.4f})")
                self.best_loss = test_loss
                self.is_best = True

            self.save_checkpoint(epoch, is_best=self.is_best)

            # --- 3. 更新学习率 (Scheduler Step) ---
            # 位置：在一个 Epoch 跑完之后
            if self.scheduler is not None:
                self.scheduler.step()
                
                # [可选] 记录当前的学习率到日志/WandB，这对调试非常有用
                # get_last_lr() 返回的是一个列表（因为可能有多个参数组），通常取第0个
                current_lr = self.scheduler.get_last_lr()[0]
                logger.info(f"LR updated to: {current_lr:.8f}") 
                
                if self.cfg.logger.enable:
                    wandb.log({"train/lr": current_lr, "epoch": epoch})
            
        logger.info("Training Finished.")