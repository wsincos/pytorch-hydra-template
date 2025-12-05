import logging
import wandb
import random
from src.utils.registry import CALLBACK_REGISTRY
from .base import Callback

logger = logging.getLogger(__name__)

@CALLBACK_REGISTRY.register("TranslationMonitor")
class TranslationMonitor(Callback):
    def __init__(self, num_samples=5):
        self.num_samples = num_samples
        self.fixed_indices = None
        self.history_data = []

    def on_epoch_end(self, solver):
        """
        每个 Epoch 结束时，执行一次翻译测试
        """
        logger.info("--- [Translation Demo] ---")
        
        # 1. 获取必要的组件
        # 注意：我们需要访问 test_loader 的 dataset 来获取原始数据
        dataset = solver.test_loader.dataset
        tokenizer = dataset.tokenizer
        
        # 2. 随机采样索引
        # 如果数据集太小，就取全部；否则取 num_samples 个

        if self.fixed_indices is None:
            total_len = len(dataset)
            # 如果数据不够，取全部；否则取固定数量
            sample_size = min(total_len, self.num_samples)
            # 这里的 seed 已经由 solver 设置过了，所以是确定的
            self.fixed_indices = random.sample(range(total_len), sample_size)
            logger.info(f"Fixed {sample_size} validation samples for monitoring.")
        
        src_texts = []
        tgt_texts = []
        
        # 2. 准备数据
        for idx in self.fixed_indices:
            src_tensor, tgt_tensor = dataset[idx]
            
            s_text = tokenizer.decode(src_tensor.tolist(), skip_special_tokens=True)
            t_text = tokenizer.decode(tgt_tensor.tolist(), skip_special_tokens=True)
            
            # 中文去空格
            t_text = t_text.replace(" ", "")
            
            src_texts.append(s_text)
            tgt_texts.append(t_text)


        # 4. [核心] 调用 Solver 的高层推理接口
        # 这一步会自动处理 Tokenization -> GPU -> Generate -> Decode
        pred_texts = solver.inference(src_texts)
        pred_texts = [p.replace(" ", "") for p in pred_texts]  # 去掉前后空格
        pred_texts = [p if p else "[Empty]" for p in pred_texts]  # 处理空预测
        
        # 5. 打印和记录
        columns = ["Epoch", "Source", "Target", "Prediction"]
        
        for s, t, p in zip(src_texts, tgt_texts, pred_texts):
            
            # 打印到控制台 (方便看 Log)
            logger.info(f"Src : {s}")
            logger.info(f"Tgt : {t}")
            logger.info(f"Pred: {p}")
            logger.info("-" * 20)
            
            # 收集给 WandB
            self.history_data.append([solver.epoch, s, t, p])

        # 6. 上传表格到 WandB
        if solver.cfg.logger.enable:
            # 每次都创建一个包含所有历史数据的新表格
            table = wandb.Table(columns=columns, data=self.history_data)
            wandb.log({"eval/samples_history": table}, step=solver.global_step)