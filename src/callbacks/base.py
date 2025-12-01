# src/callbacks/base.py
class Callback:
    """
    Callback 基类。
    所有具体的回调（如保存模型、早停）都要继承这个类。
    """
    def on_train_start(self, solver): pass
    def on_train_end(self, solver): pass
    
    def on_epoch_start(self, solver): pass
    def on_epoch_end(self, solver): pass
    
    def on_step_start(self, solver): pass
    def on_step_end(self, solver): pass

    def on_scheduler_step(self, solver): pass