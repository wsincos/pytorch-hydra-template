import torch
import torch.nn as nn
from src.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class Seq2SeqWrapper(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output
    
    @torch.no_grad()
    def generate(self, src, bos_token_id, eos_token_id, max_len=128):
        """
        底层生成函数：输入 Tensor，输出 Tensor
        Args:
            src: [Batch, SeqLen] 输入的 Token IDs
            bos_token_id: int
            eos_token_id: int
        Returns:
            generated_seqs: [Batch, OutSeqLen]
        """
        was_training = self.training
        if was_training:
            self.eval()
        device = src.device
        batch_size = src.size(0)

        # 1. Encoder 编码 -> Memory: [Batch, SeqLen, Dim]
        memory = self.encoder(src)

        # 2. 准备 Decoder 的起始输入: [Batch, 1] 全是 BOS
        ys = torch.full((batch_size, 1), bos_token_id, dtype=torch.long).to(device)
        
        # 记录每个样本是否已经结束 (遇到了 EOS)
        finished = torch.zeros(batch_size, dtype=torch.bool).to(device)

        # 3. 自回归循环
        # 使用 top-k 采样替代纯贪婪搜索，减少重复高频token（例如标点或常用单字）
        top_k = 5
        temperature = 1.0
        for _ in range(max_len):
            # 将当前已生成的序列送入 Decoder
            out = self.decoder(ys, memory)
            # 取最后一个时间步的logits: [Batch, VocabSize]
            logits = out[:, -1]

            # 温度缩放
            if temperature != 1.0:
                logits = logits / temperature

            # top-k 采样
            if top_k > 0:
                topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=1)
                probs = torch.softmax(topk_vals, dim=1)
                # 从 top-k 分布中采样一个词
                sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
                next_word = topk_idx.gather(1, sampled.unsqueeze(1)).squeeze(1)
            else:
                # 退回到贪婪策略
                _, next_word = torch.max(logits, dim=1)

            # 如果某个样本已经结束了，就让它一直生成 EOS (作为 Padding)
            next_word = torch.where(finished, torch.tensor(eos_token_id).to(device), next_word)

            # 更新结束状态
            finished |= (next_word == eos_token_id)

            # 拼接到结果后面: [Batch, CurrLen + 1]
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)

            # 如果所有样本都结束了，提前退出
            if finished.all():
                break
        
        
        if was_training:
            self.train()
            
        # 返回结果 (去掉第一列的 BOS)
        return ys[:, 1:]
    