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
        sample_size = min(len(dataset), self.num_samples)
        indices = random.sample(range(len(dataset)), sample_size)
        
        src_texts = []
        tgt_texts = []
        
        # 3. 准备数据：从 Tensor ID 还原回 字符串 Text
        # 因为 solver.inference 接收的是字符串列表
        for idx in indices:
            src_tensor, tgt_tensor = dataset[idx]
            
            # decode: 将 [101, 234, 102] -> "Hello world"
            s_text = tokenizer.decode(src_tensor.tolist(), skip_special_tokens=True)
            t_text = tokenizer.decode(tgt_tensor.tolist(), skip_special_tokens=True)
            
            # (可选) 去掉 BERT Tokenizer 对中文可能产生的空格，让显示更自然
            t_text = t_text.replace(" ", "")
            
            src_texts.append(s_text)
            tgt_texts.append(t_text)

        # 4. [核心] 调用 Solver 的高层推理接口
        # 这一步会自动处理 Tokenization -> GPU -> Generate -> Decode
        pred_texts = solver.inference(src_texts)
        pred_texts = [p.replace(" ", "") for p in pred_texts]  # 去掉前后空格
        pred_texts = [p if p else "[Empty]" for p in pred_texts]  # 处理空预测
        
        # 5. 打印和记录
        columns = ["Source", "Target", "Prediction"]
        wandb_data = []
        
        for s, t, p in zip(src_texts, tgt_texts, pred_texts):
            
            # 打印到控制台 (方便看 Log)
            logger.info(f"Src : {s}")
            logger.info(f"Tgt : {t}")
            logger.info(f"Pred: {p}")
            logger.info("-" * 20)
            
            # 收集给 WandB
            wandb_data.append([s, t, p])

        # 6. 上传表格到 WandB
        if solver.cfg.logger.enable:
            table = wandb.Table(columns=columns, data=wandb_data)
            # 使用 epoch 作为 key 可能会覆盖，建议加上 step 或者用特定的 key
            wandb.log({"eval/samples": table}, step=solver.global_step)