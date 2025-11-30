"""
datasets.py的内容:

1. get_tokenizer: 获取预训练模型的Tokenizer单例。
2. Opus100Dataset: 用于加载本地.parquet格式的翻译数据集。
3. get_dataloader: 加载和处理翻译数据集的模块。

"""

import os
import logging

import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import AutoTokenizer
from src.utils.registry import DATASET_REGISTRY

logger = logging.getLogger(__name__)

# 全局缓存 Tokenizer
_tokenizer = None

def get_tokenizer(model_path=None):
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    
    if model_path is None:
        model_path = "google-bert/bert-base-chinese"
    
    logger.info(f"[Data] Loading Tokenizer from: {model_path}...")
    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
    except OSError:
        logger.warning(f"[Data] Local path not found. Downloading...")
        _tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
        
    return _tokenizer


@DATASET_REGISTRY.register()
class Opus100Dataset(Dataset):
    def __init__(self, cfg, split='train'):
        self.cfg = cfg

        if split == "train":
            parquet_path = cfg.dataset.train_path
        elif split == "test":
            parquet_path = cfg.dataset.test_path
        else:
            raise ValueError(f"Unsupported split: {split}")

        # 1. 检查文件是否存在
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"training file not found at: {parquet_path}")
        
        logger.info(f"[Data] Loading local parquet file: {parquet_path}...")

        # 2. 加载 Parquet

        ## 用huggingface的datasets里面的load_dataset来加载本地parquet文件
        ## data_files 参数支持直接读取本地文件
        try:
            dataset_dict = load_dataset("parquet", data_files={split: parquet_path}, split=split)
        except Exception as e:
            raise RuntimeError(f"Failed to load parquet file. Error: {e}")
        self.data = dataset_dict

        ## 如果无法使用load_dataset, 可以使用pandas直接读取
        ## engine='pyarrow' 是最常用的引擎，速度快
        # try:
        #     df = pd.read_parquet(parquet_path, engine='pyarrow')
        # except Exception as e:
        #     raise RuntimeError(f"Failed to load parquet file using Pandas. Error: {e}")
        # self.data = df.to_dict('records')



        # 3. 限制数据量（用于快速测试）
        if split == 'train' and cfg.dataset.max_samples > 0:
            logger.info(f"[Data] Selecting first {cfg.dataset.max_samples} samples...")
            self.data = self.data.select(range(min(len(self.data), cfg.dataset.max_samples)))
            
        self.tokenizer = get_tokenizer(cfg.dataset.tokenizer_path)
        self.src_lang = cfg.dataset.source_lang
        self.tgt_lang = cfg.dataset.target_lang
        self.max_len = cfg.dataset.seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # OPUS-100 的 Parquet 结构是: {'translation': {'en': '...', 'zh': '...'}}
        item = self.data[idx]
        
        # 提取源语言和目标语言文本
        src_text = item['translation'][self.src_lang]
        tgt_text = item['translation'][self.tgt_lang]

        # Tokenization
        src_tokens = self.tokenizer(
            src_text, truncation=True, max_length=self.max_len, return_tensors=None
        )['input_ids']

        tgt_tokens = self.tokenizer(
            tgt_text, truncation=True, max_length=self.max_len, return_tensors=None
        )['input_ids']

        src_tokens = torch.tensor(src_tokens, dtype=torch.long)
        tgt_tokens = torch.tensor(tgt_tokens, dtype=torch.long)

        return src_tokens, tgt_tokens
        

    def get_vocab_size(self):
        return self.tokenizer.vocab_size

# === Collate Function ===
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    tokenizer = get_tokenizer()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)
    return src_padded, tgt_padded

# === Loader Builder ===
def get_dataloader(cfg, split='train'):
    dataset_cls = DATASET_REGISTRY.get(cfg.dataset.name)
    
    # 实例化时把 split 传进去
    dataset = dataset_cls(cfg, split=split)
    
    # 训练集：必须 Shuffle (打乱)，否则模型学不到东西
    # 测试集：不要 Shuffle，方便对比结果
    is_shuffle = (split == 'train')
    
    return DataLoader(
        dataset, 
        batch_size=cfg.dataset.batch_size, 
        shuffle=is_shuffle, 
        num_workers=cfg.dataset.num_workers,
        collate_fn=collate_fn
    )