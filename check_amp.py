import torch
from omegaconf import OmegaConf
from hydra import initialize, compose

# 加载配置
with initialize(config_path="conf", version_base=None):
    cfg = compose(config_name="config")

print("=" * 50)
print("AMP 配置检查")
print("=" * 50)
print(f"use_amp 设置: {cfg.train.get('use_amp', False)}")
print(f"CUDA 可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    capability = torch.cuda.get_device_capability()
    print(f"计算能力: {capability[0]}.{capability[1]}")
    
    if capability[0] >= 7:
        print("✓ GPU 支持 Tensor Core，AMP 效果最佳")
    else:
        print("⚠️ GPU 不支持 Tensor Core，AMP 效果有限")

print("\n" + "=" * 50)
print("测试 AMP 显存节省效果")
print("=" * 50)

if torch.cuda.is_available():
    device = torch.device('cuda')
    
    # 简单模型测试
    model = torch.nn.TransformerEncoderLayer(d_model=512, nhead=8).to(device)
    
    # FP32 测试
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    x = torch.randn(96, 64, 512).to(device)  # batch_size=96
    with torch.no_grad():
        out = model(x)
    
    fp32_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"FP32 显存峰值: {fp32_mem:.2f} MB")
    
    # FP16 测试
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            out = model(x)
    
    fp16_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"AMP 显存峰值: {fp16_mem:.2f} MB")
    print(f"节省: {(1 - fp16_mem/fp32_mem)*100:.1f}%")
