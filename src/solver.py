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
from torch.cuda.amp import autocast, GradScaler

import src.callbacks
from src.utils.common import dict_from_config
from src.models.builders import get_model
from src.utils.common import register_standard_components, seed_everything
from src.data.datasets import get_dataloader
from src.utils.registry import (OPTIMIZER_REGISTRY, 
                                CRITERION_REGISTRY, 
                                SCHEDULER_REGISTRY, 
                                CALLBACK_REGISTRY)

logger = logging.getLogger(__name__)

class Solver:
    def __init__(self, cfg):
        self.cfg = cfg
        seed_everything(cfg.train.seed)
        register_standard_components()
        self.device = torch.device(cfg.train.device)

        # 混合精度训练
        self.use_amp = cfg.train.get('use_amp', False) if torch.cuda.is_available() else False
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("Mixed Precision Training (AMP) enabled.")

        # 状态变量
        self.epoch = 0
        self.global_step = 0
        self.current_loss = 0.0
        self.best_loss = float('inf')
        self.train_loss = float('inf') # 供 Callback 读取
        self.test_loss = float('inf')  # 供 Callback 读取
        # self.is_best = False         # 供 Callback 读取
        self.should_stop = False       # 供 Callback 修改

        # 获取 Hydra 输出目录 (用于给 CheckpointCallback 传参)
        try:
            self.run_dir = HydraConfig.get().runtime.output_dir
        except:
            self.run_dir = "."


        # === 1. 数据 ===
        logger.info("Building Loader...")
        self.train_loader = get_dataloader(cfg, split='train')
        self.test_loader = get_dataloader(cfg, split='test')


        # 自动调整模型词表大小 (适配 BERT)
        if hasattr(self.train_loader.dataset, 'get_vocab_size'):
            vocab_size = self.train_loader.dataset.get_vocab_size()
            logger.info(f"Auto-setting vocab size from Tokenizer: {vocab_size}")
            
            cfg.model.shared.src_vocab_size = vocab_size
            cfg.model.shared.tgt_vocab_size = vocab_size

        
        # === 2. 模型、优化器、Loss、Scheduler ===
        logger.info(f"Building Model: {cfg.model.name}")
        self.model = get_model(cfg).to(self.device)
        
        optim_cls = OPTIMIZER_REGISTRY.get(cfg.optimizer.name)
        self.optimizer = optim_cls(self.model.parameters(), **cfg.optimizer.params)
        
        loss_cls = CRITERION_REGISTRY.get(cfg.criterion.name)
        self.criterion = loss_cls(ignore_index=0, **cfg.criterion.params)
        
        self.scheduler = None
        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * self.cfg.train.epochs

        if "scheduler" in cfg and cfg.scheduler is not None:
            sched_cls = SCHEDULER_REGISTRY.get(cfg.scheduler.name)
            if cfg.scheduler.name == "OneCycleLR":
                cfg.scheduler.params["total_steps"] = total_steps
            self.scheduler = sched_cls(self.optimizer, **(cfg.scheduler.params or {}))
        

        # === 3. 构建 Callback 系统 ===
        self.callbacks = []
        if "callback" in cfg and cfg.callback is not None:
            # 遍历配置中的每个 callback
            for cb_key, cb_conf in cfg.callback.items():
                logger.info(f"Building Callback: {cb_key} ({cb_conf.name})")
                
                # 获取参数字典
                params = dict_from_config(cb_conf.params) if cb_conf.params else {}
                
                # [特殊逻辑] 如果是 CheckpointCallback 且没指定路径，自动注入 Hydra 路径
                if cb_conf.name == "CheckpointCallback" and "save_dir" not in params:
                    params['save_dir'] = os.path.join(self.run_dir, "checkpoints")
                
                # 实例化并加入列表
                cb_cls = CALLBACK_REGISTRY.get(cb_conf.name)
                self.callbacks.append(cb_cls(**params))

        # === 4. WandB ===
        if cfg.logger.enable:
            wandb_cfg = OmegaConf.to_container(cfg.logger, resolve=True)
            wandb_cfg.pop('enable')
            wandb.init(dir=self.run_dir, **wandb_cfg, config=OmegaConf.to_container(cfg, resolve=True))

    def trigger_callbacks(self, hook_name):
        """统一触发所有 Callback 的钩子函数"""
        for cb in self.callbacks:
            # 相当于调用 cb.on_epoch_end(self)
            getattr(cb, hook_name)(self)
    

    @torch.no_grad()
    def inference(self, text_list):
        """
        输入文本列表，输出翻译结果列表。
        负责处理分词 (Tokenization) 和解码 (Decoding)。
        
        Args:
            text_list: list[str], 例如 ["Hello world", "Deep learning"]
        Returns:
            list[str], 例如 ["你好世界", "深度学习"]
        """
        self.model.eval()
        
        # 1. 获取 Tokenizer (从训练集 Dataset 中借用)
        # 假设 Dataset 里已经初始化了 self.tokenizer
        tokenizer = self.train_loader.dataset.tokenizer

        
        
        # 获取特殊 Token ID (用于生成时的控制)
        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id  # BERT 的 BOS 是 [CLS]
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id  # BERT 的 EOS 是 [SEP]
        
        # 2. 文本 -> Tensor (Batch Tokenization)
        # padding=True: 自动把这一批文本补齐到同样长度
        # truncation=True: 防止输入太长爆显存
        inputs = tokenizer(
            text_list, 
            padding=True, 
            truncation=True,    
            max_length=128, 
            return_tensors=None
        )
        
        src_tensor = torch.tensor(inputs['input_ids']).to(self.device) # [Batch, SeqLen]
        
        # 3. 调用模型的底层 generate
        generated_ids = self.model.generate(
            src_tensor, 
            bos_token_id=bos_id, 
            eos_token_id=eos_id,
            max_len=64
        )
        
        # 4. Tensor -> 文本 (Batch Decoding)
        # skip_special_tokens=True 会自动去掉 [CLS], [SEP], [PAD]
        decoded_preds = tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        
        self.model.train()
        return decoded_preds
            
    def run_step(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        tgt_input = y[:, :-1]
        tgt_label = y[:, 1:]
        
        # 使用混合精度前向传播
        with autocast(enabled=self.use_amp):
            logits = self.model(x, tgt_input)
            loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_label.reshape(-1))
        
        if self.model.training:
            self.optimizer.zero_grad()
            
            if self.use_amp:
                # 混合精度反向传播和优化器更新
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 标准反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            self.global_step += 1

        return loss.item()
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        for i, (x, y) in enumerate(self.test_loader):
            loss = self.run_step(x, y)
            total_loss += loss
        avg_loss = total_loss / len(self.test_loader)
        
        # if self.cfg.logger.enable:
        #     wandb.log({"test/loss": avg_loss}, step=self.epoch)
        
        # 更新状态供 Callback 读取
        self.test_loss = avg_loss
        self.model.train()
        return avg_loss


    def train(self):
        logger.info(f"Start Training... Total Epochs: {self.cfg.train.epochs}")
        self.model.train()
        
        # 触发开始钩子
        self.trigger_callbacks("on_train_start")
        
        for epoch in range(self.cfg.train.epochs):
            self.epoch = epoch
            self.trigger_callbacks("on_epoch_start")
            
            # --- 训练循环 ---
            total_loss = 0
            for i, (x, y) in enumerate(self.train_loader):
                loss = self.run_step(x, y)
                total_loss += loss
                self.current_loss = loss

                self.trigger_callbacks("on_step_end")
                # if i % 100000 == 0:
                #     logger.info(f"Epoch {epoch} | Step {i} | Train Loss: {loss:.4f}")
                #     if self.cfg.logger.enable:
                #         wandb.log({"train/step_loss": loss})

                if self.scheduler is not None and self.cfg.scheduler.name == "OneCycleLR":
                    self.scheduler.step()
                    # 触发 Monitor 记录 LR (变成 Step 级记录，曲线更平滑)
                    self.trigger_callbacks("on_scheduler_step")
            
            avg_train_loss = total_loss / len(self.train_loader)
            self.train_loss = avg_train_loss
            # logger.info(f"=== Epoch {epoch} Done | Train Loss: {avg_train_loss:.4f} ===")
            # if self.cfg.logger.enable:
            #     wandb.log({"train/epoch_loss": avg_train_loss, "epoch": epoch})
            
            
            # --- 评估 ---
            self.evaluate() # 内部更新了 self.test_loss
            
            
            # --- 触发 Epoch 结束回调 ---
            # 这里会执行：保存模型、检查早停
            self.trigger_callbacks("on_epoch_end")
            
            # --- 检查是否早停 ---
            if self.should_stop:
                logger.info("Training stopped by Callback.")
                break

            # --- 学习率更新 ---
            if self.scheduler is not None and self.cfg.scheduler.name != "OneCycleLR":
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]
                logger.info(f"LR updated to: {lr:.8f}")
                # if self.cfg.logger.enable:
                #     wandb.log({"train/lr": lr, "epoch": epoch})
                self.trigger_callbacks("on_scheduler_step")
            
        self.trigger_callbacks("on_train_end")
        logger.info("Training Finished.")