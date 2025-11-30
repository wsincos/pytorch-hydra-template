import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from src.solver import Solver

# 初始化 Logger
# 在 Hydra 启动后，这个 logger 会自动继承 Hydra 的日志配置
# 输出会同时显示在屏幕上和保存到 .log 文件中
logger = logging.getLogger(__name__)

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    
    # 使用 logger 打印配置信息
    # 这样 log 文件里开头就是详细的参数列表，非常方便复盘
    logger.info("====== Configuration ======")
    
    # resolve=True 很重要，它会把 ${model.shared.d_model} 这种变量解析成具体的数字
    logger.info("\n" + OmegaConf.to_yaml(cfg, resolve=True))
    
    logger.info("===========================")
    
    # 3. 实例化求解器
    logger.info(f"Initializing Solver with GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All')}")
    solver = Solver(cfg)
    
    # 4. 开始训练
    solver.train()

if __name__ == "__main__":
    main()