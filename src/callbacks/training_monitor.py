import wandb
import time
from src.utils.registry import CALLBACK_REGISTRY
from .base import Callback
import logging

logger = logging.getLogger(__name__)

@CALLBACK_REGISTRY.register("TrainingMonitor")
class TrainingMonitor(Callback):
    def __init__(self, log_every_n_steps=100000):
        self.log_every = log_every_n_steps
        self.start_time = time.time()

    def on_step_end(self, solver):
        if solver.cfg.logger.enable and solver.global_step % self.log_every == 0:
            
            loss = getattr(solver, "current_loss", None)
            if loss is not None:
                wandb.log({"train/loss": loss}, step=solver.global_step)
    
    def on_epoch_end(self, solver):
        if solver.cfg.logger.enable:
            wandb.log({"train/epoch_loss": solver.train_loss}, step=solver.epoch)

            test_loss = getattr(solver, "test_loss", None)
            if test_loss is not None:
                wandb.log({"test/epoch_loss": test_loss}, step=solver.epoch)

    def on_scheduler_step(self, solver):
        if solver.cfg.logger.enable:
            lr = solver.scheduler.get_last_lr()[0] if solver.scheduler else 0.0
            wandb.log({"train/lr": lr}, step=solver.global_step)