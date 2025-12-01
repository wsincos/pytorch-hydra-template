# src/callbacks/early_stopping.py
import logging
from src.utils.registry import CALLBACK_REGISTRY
from .base import Callback

logger = logging.getLogger(__name__)

@CALLBACK_REGISTRY.register("EarlyStopping")
class EarlyStoppingCallback(Callback):
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def on_epoch_end(self, solver):
        # 从 solver 获取当前的测试集 Loss
        current_loss = getattr(solver, "test_loss", None)
        
        if current_loss is None:
            return

        if current_loss < self.min_validation_loss:
            self.min_validation_loss = current_loss
            self.counter = 0
        elif current_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            logger.info(f"[EarlyStopping] Counter: {self.counter}/{self.patience} (Best: {self.min_validation_loss:.4f})")
            
            if self.counter >= self.patience:
                logger.info(f"Early Stopping triggered!")
                # 修改 Solver 的状态，通知它停车
                solver.should_stop = True