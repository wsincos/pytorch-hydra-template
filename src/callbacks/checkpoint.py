# src/callbacks/checkpoint.py
import os
import torch
import logging
from omegaconf import OmegaConf
from src.utils.registry import CALLBACK_REGISTRY
from .base import Callback

logger = logging.getLogger(__name__)

@CALLBACK_REGISTRY.register("CheckpointCallback")
class CheckpointCallback(Callback):
    def __init__(self, save_dir, save_every=1, keep_last=True):
        self.save_dir = save_dir
        self.save_every = save_every
        self.keep_last = keep_last
        
        # 确保保存目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"[Callback] Checkpoints will be saved to: {self.save_dir}")

    def on_epoch_end(self, solver):
        """每个 Epoch 结束时触发保存逻辑"""
        epoch = solver.epoch
        
        current_loss = getattr(solver, "test_loss", float('inf'))
        is_best = False
        if current_loss < solver.best_loss:
            logger.info(f"New Best Model! Loss: {solver.best_loss:.4f} -> {current_loss:.4f}")
            solver.best_loss = current_loss
            is_best = True

        # 从 Solver 获取状态
        state = {
            "epoch": epoch,
            "model_state": solver.model.state_dict(),
            "optimizer_state": solver.optimizer.state_dict(),
            # 判空处理
            "scheduler_state": solver.scheduler.state_dict() if solver.scheduler else None,
            "best_loss": solver.best_loss,
            "config": OmegaConf.to_container(solver.cfg, resolve=True)
        }

        # 1. 保存 last.pt (始终覆盖，用于断点续训)
        if self.keep_last:
            last_path = os.path.join(self.save_dir, "checkpoint_last.pt")
            torch.save(state, last_path)

        # # 2. 按频率保存历史存档
        # if (epoch + 1) % self.save_every == 0:
        #     path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        #     torch.save(state, path)
        #     logger.info(f"Saved checkpoint: {path}")

        # 3. 保存最佳模型 (依赖 solver.is_best 标志位)
        
        if is_best:
            best_path = os.path.join(self.save_dir, "checkpoint_best.pt")
            torch.save(state, best_path)
            logger.info(f"Saved BEST model to {best_path}")