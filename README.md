# PyTorch Hydra Template (Meta-Style)

这个仓库的目的是为了讲解如何构建一套`hydra-style`的标准化工作流。一个基于Meta/FAIR工程实践的强大、配置驱动的深度学习项目模板。

论文/项目级别的代码，往往是一个工程量极大、代码文件之间关联极深的项目，如果不对代码的**编写方式和风格**加以控制，那么随着代码量和文件的累积，编写代码和debug将变得越来越力不从心。因此，代码需要有一套统一化规范化的编写范式和模板。

该模板具有用于配置管理的 Hydra、用于模块化的 Registry Pattern 和用于灵活构建模型的 Builder Pattern。其设计具有高度可扩展性，适用于大规模消融研究。



本仓库以`Transformer seq2seq task`任务为例，讲述**从项目创建到发布的完整流程**，包括 `github`仓库管理、代码编写、文件管理等内容



## 0.创建仓库

### 0.1 创建github仓库

在明确项目目标以后，我们可以首先创建一个github仓库，后续将更新的内容都同步到这里。



**创立项目文件夹并与仓库连接：**

在仓库创建以后，我们可以创建一个项目所在具体文件夹，并将git仓库与文件夹同步

一般流程如下首先到对应的文件夹，然后执行以下指令

```bash
git init

# 创建readme
echo "#{readme_title}" >> README.md

# 这一步会创建一个基本的 Python 忽略文件，如果已有可跳过
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".DS_Store" >> .gitignore

# 将内容加入缓存区
git add .

git commit -m "first commit" # 提交到本地历史记录
git branch -M main # 把当前分支重命名为 main
# 给远程仓库的地址起个别名，叫 origin
git remote add origin <https://github.com/user_name/project_name.git>
git push -u origin main # 把代码推送到 origin，并确立“绑定”关系
```



### 0.2 初始化工作环境

还有一些别的项目初始化相关内容，包括如下

-   创建新的虚拟环境

    ```bash
    conda create -n new_env python=3.10
    ```

-   添加datasets和pretrained_models（即`1.项目结构`中data文件夹）

    >    在放入数据以后，将对应的文件加入到.gitignore中

## 1. 项目结构

一个**configuration-driven**的项目，其基本结构可以分成几个板块

-   conf：存放所有配置和具体值
-   src：存放所有的代码，包括模型、数据处理、utils、`sovler`
-   data：存放所有数据集、pretrained_models
-   train.py
-   requirements.txt
-   outputs



具体一个项目结构的**创建例子**如下：

```plain text
.
├── conf/                       # [Configuration Center]
│   ├── config.yaml             # Main entry config
│   ├── model/                  # Model architecture configs
│   ├── dataset/                # Dataset paths & params
│   ├── optimizer/              # Optimizer configs
│   └── logger/                 # WandB settings
├── src/                        # [Source Code]
│   ├── utils/
│   │   ├── registry.py         # The core Registry implementation
│   │   └── common.py           # PyTorch standard component registration
│   ├── models/
│   │   ├── components.py       # Encoder/Decoder definitions
│   │   └── builders.py         # Logic to assemble models from config
│   ├── data/
│   │   └── datasets.py         # Dataset classes (Parquet/HF support)
│   └── solver.py               # The main training loop controller
├── data/                       # [Data Assets] (Git ignored)
│   ├── datasets/               # Raw data
│   ├── pretrained_models/      # Tokenizers & Pretrained weights
├── train.py                    # Entry point
└── requirements.txt            # Dependencies
```



具体来说，创建工作目录可以通过下面的方法实现：

```bash
# 1. 创建config目录
mkdir -p ./conf/{model,optimizer,dataset,logger}
touch ./conf/config.yaml

# 2. 创建source目录
mkdir -p ./src/{utils,models,data}
touch ./src/solver.py

# 3. 创建data目录
mkdir -p ./data/{datasets,pretrained_models}

# 4. train.py
touch train.py

# 5. requirements.py
# touch requirements.txt # 如果没有requirements，也可以不创建

```

>   注意我们这里并没有创建outputs文件夹，因为这个是hydra在运行结束以后会自行创建的，所以我们可以不用创建



## 2. 编写utils

### 2.1 编写Registry (src/utils/registry.py)

基于 `Hydra-style` 的项目代码，其中一大特征就是使用了 **Registry (注册器)** 模式。

**Registry 是一个将“字符串名称”映射到“Python 类或函数”的全局容器。** 它充当了**静态配置文件 (YAML)** 与 **动态代码逻辑** 之间的桥梁。

具体来说，我们在YAML中存放的是一系列的“变量”，是以字符的形式呈现的，但是在代码中，这些“变量”都是在类里面使用的，用于**初始化类**，而**Registry则存放了所有的类**，包括Model、Optimizer、Dataset等所有类。后续我们只需要将类从Registry取出来，用这些“变量”即可创建一个满足我们条件的“组件”。

在代码中，我们在创建一个类的时候，会利用`@register` 装饰器的方式，把每一个类都放入注册表中，确保Registry中包含了所有的类。



**实现**

我们将`registry.py`文件写在`src/utils`文件夹之中

```python
class Registry:
    def __init__(self, name):
        pass

    def registrer(self, obj_name):
        pass

    def get(self, obj_name):
        pass

MODEL_REGISTRY = Registry("MODEL")
...
```



### 2.2 通用工具模块（src/utils/common.py）

我们在2中提到过，在创建类的时候，会用装饰器把每一个类放入注册表，但是对于AdamW等本身已经在torch中实现的类，我们不需要再创建，但是依然需要将其加入注册表中。因此我们需要一个函数，将这些已经存在的类加入到注册表中。

由于这个函数与我们整个项目的关系不大，因此我们将其写在`src/utils/common.py`中



同时我们还可以在`src/utils/common.py`中加入一些别的函数，例如`seed_everything`等

```python
def seed_everything(seed):
    pass
    
def register_standard_components():
    pass
```





## 3.模型各个组件定义

我们本次实验用的是Transformer来做翻译任务，为了体现模块化的想法，我们将整个模型分为`Encoder`和`Decoder`两个模块，用一个`Seq2seqWrapper`包装在一起。

上述所有的模块，通过`builder.py`进行组装



### 3.1 创建modules

-   layers.py

    ```python
    class PositionalEncoding(nn.Module):
        ...
    ```

    这里我们没有对`PositionalEncoding`注册，因为我们默认使用最简单的

-   encoder.py

    ```python
    @ENCODER_REGISTRY.register("TransformerEncoder")
    class MyEncoder(nn.Module):
        def __init__(self, input_dim, d_model, nhead, num_layers, pe_config, dropout=0.1):
            ...
            
        def forward(self, x):
            ...
    ```

-   decoder.py

    ```python
    @DECODER_REGISTRY.register("TransformerDecoder")
    class MyDecoder(nn.Module):
        def __init__(self, output_dim, d_model, nhead, num_layers, pe_config, dropout=0.1):
            ...
            
        def forward(self, tgt, memory):
            ...
    ```

-   wrappers.py

    ```python
    @MODEL_REGISTRY.register()
    class Seq2SeqWrapper(nn.Module):
        def __init__(self, encoder, decoder):
            ...
    
        def forward(self, src, tgt):
            ...
    ```



在创建encoder和decoder的时候，只需要考虑输入和输出的变量有哪些，而不需要考虑任何值，只需要编写模型书写逻辑即可。需要各种config的时候，我们再在conf文件夹对应位置找



wrappers.py是一个“胶水层”，encoder和decoder只是定义了模型的不同组件，我们现在要做的事情就是将他们“黏合”到一起，通过wrapper实现。

wrapper在实现的时候，完全不用纠结encoder、decoder具体是什么，我们只需要关注我们的数据`src`和`tgt`怎么输入到这些组件里面去即可，这样又进一步将数据“抽象化”。





### 3.2 实例化Model

我们通过`src/models/builder.py`来实例化一个具体的Transformer，包括各种参数，实现后续代码可以使用`builder.py`得到完整model并直接使用。



#### 3.2.1 编写build.py

>   整个`models` 文件夹中 只有build.py中的`get_model`函数是暴露的。

我们依然先不纠结具体的参数值，而是**假设所有需要的参数都存储在一个`omegaconf.DictConfig`类型的变量`cfg`中**。而我们要使用的则是cfg.model中的内容

cfg.model可以分成多个逻辑区块

-   身份区：name（决定builder进入哪个if分支）、wrapper（确定实例化的容器类，如果有别的参数，这个也可以作为一个单独的逻辑分区）

-   共享区：shared（全局变量），这是为了遵循 **DRY (Don't Repeat Yourself)** 原则。

    >   e.g.Encoder 和 Decoder 通常必须拥有相同的 hidden dimension (`d_model`) 才能对接。如果在 `arch` 里分别写两遍 `128`，当你做消融实验想改成 `256` 时，很容易只改了一个而忘了另一个，导致报错。在这里定义一次，通过 `${model.shared.d_model}` 引用，安全又方便。

-   架构区：arch（包括`encoder: {...}`、`decoder: {...}`）

-   初始化策略、正则化配置、归一化配置、预训练加载等等

```python
"""
读取 Config 并实例化Transformer模型组件的模块。
"""
from curses import wrapper
import logging
import omegaconf
from src.utils.registry import (ENCODER_REGISTRY, DECODER_REGISTRY, MODEL_REGISTRY)
from src.utils.common import dict_from_config

logger = logging.getLogger(__name__)

def get_model(cfg: omegaconf.DictConfig):
    """
    根据配置实例化Transformer模型组件。

    参数:
        cfg (omegaconf.DictConfig): 包含模型组件配置的字典。

    返回:
        model: 实例化的Transformer模型。
    """
    model_name = cfg.model.name
    
    if model_name == 'seq2seq_transformer':
        # 1. 提取框架配置(所有组件的配置都在其中)
        arch_cfg = dict_from_config(cfg.model.arch)
        
        # 2. 分离配置
        encoder_cfg = arch_cfg.pop("encoder")
        decoder_cfg = arch_cfg.pop("decoder")

        # 3. 实例化组件
        encoder_name = encoder_cfg.pop("name")
        logger.info(f"[Builder] Instantiating Encoder: {encoder_name}")
        encoder = ENCODER_REGISTRY.get(encoder_name)(**encoder_cfg)

        decoder_name = decoder_cfg.pop("name")
        logger.info(f"[Builder] Instantiating Decoder: {decoder_name}")
        decoder = DECODER_REGISTRY.get(decoder_name)(**decoder_cfg)

        # 4. 实例化模型
        wrapper_cfg = dict_from_config(cfg.model.wrapper)
        wrapper_name = wrapper_cfg.pop("name")
        logger.info(f"[Builder] Instantiating Model: {wrapper_name}")
        wrapper_cls = MODEL_REGISTRY.get(wrapper_name)
        return wrapper_cls(encoder=encoder, decoder=decoder, **wrapper_cfg)
    else:
        raise KeyError(f"Unknown model name: {model_name}")
```

解释：

-   我们需要利用config文件来实例化模型，所以我们只需要向函数中传入config文件

-   由于build函数是一个高度抽象化的函数，后续可能不只是我们当前的transformer model，还会涉及到一些别的模型的build（比如Unet等各种架构），所以我们需要给这个transformer model一个单独的名字`seq2seq_transformer`

-   这里之所以用pop的原因，是为了方便后续字典解包。

    我们在编写config的时候，除了写传入组件对应的各个参数值以外，我们额外还会给一个name，所以如果直接解包，那么将导致模型中出现无法理解的参数

-   既然这里涉及到实例化了，那必然就会涉及到具体的参数，所以我们**需要设定模型相关的参数**。基于上述实验中用到的各种参数，我们可以得到如下的一个`transformer.yaml`放在`conf/models`中

    -   model_name
    -   wrapper
    -   arch
        -   encoder : {name, input_dim, d_model, nhead, num_layers, pe_cfg} # 这些是model实例化时需要的参数
        -   decoder : {name, output_dim, d_model, nhead, num_layers, pe_cfg}



### 3.3 细节处理

由于python运行程序时，它**不会**自动扫描项目里的每一个 `.py` 文件，它只会加载你**显式导入（Import）** 的文件。

而**整个`models`文件夹暴露在外的只有一个函数`get_model`**。由于 `src/models/modules/encoders.py` 这个文件**从来没有被 Python 读取过**。 既然文件没被读取，里面的 `@register` 装饰器代码就没有机会运行。

当`train.py`或其他文件使用`get_model`函数时，`get_model` 去 `ENCODER_REGISTRY` 字典里查 `"TransformerEncoder"` 这个名字，则为空。

所以我们需要在`src/models/__init__.py`中，导入这些含有注册表的文件

```python
from .modules import (encoders, decoders, 
                      wrappers, layers)

from .builders import get_model

# 暴露给外部方便调用
__all__ = ["get_model"]
```





## 4. 数据处理

我们通过`src/data/datasets.py`对数据进行处理

 在编写数据相关的内容时，抓住一条核心的线：

**核心目标是得到一个`dataloader`** 

```python
def get_dataloader(cfg, split):
  
	pass

  return DataLoader(
          dataset, 
          batch_size=cfg.dataset.batch_size, 
          shuffle=is_shuffle, 
          num_workers=cfg.dataset.num_workers,
          collate_fn=collate_fn
      )
```

即得到这样的一个东西。其中`batch_size`、`shuffle`、`num_workers`都是config中需要设定的内容，而`dataset`是dataloader的核心，`collate_fn`是用于处理这些数据的SeqLen不一致的情况

所以现在的目标是两个：

1.   编写dataset并实例化

     编写dataset的核心是实现`__len__`函数、`__getitem__`函数以及`get_vocab_size`函数(NLP任务)。然后基于实现上述的内容的目的，在`__init__`中处理数据和编写变量

2.   编写collate_fn函数

     写清楚如何如何对batch中的数据做处理。



基于上述的代码，我们再完成`./conf/dataset/opus_enzh.yaml`的编写



## 5.编写solver.py

`./src/solver.py`是实验的核心控制器，其核心是为了实现`train`

`train`整个工作流可以分为如下内容：

-   `dataloader`遍历数据
-   模型运行一次并计算loss
-   保存模型state
-   日志保存

### 5.1 callback
Callback 是什么：是插入到主程序流程中的**“钩子（Hooks）”或“插件（Plugins）”**。

如何理解：它是**“事件驱动”**的。当 Solver 触发某个事件（如 Epoch 结束）时，所有订阅了这个事件的 Callback 就会自动运行。

意义：实现了关注点分离。Solver 负责“跑”，Callback 负责“看”和“记”。这是构建大型、易维护系统的基石。
具体来说，callback将训练过程中需要处理的各项事宜**分离**出train函数，实现解耦。
例如`save_checkpoint`, `early_stop`, `wandb_send`等操作均由callback完成，只需在train函数加上`self.trigger_callbacks("on_epoch_end")`即可

判断一个函数是否应该放入callback中：如果删掉这段代码，模型的训练结果（权重数值）会变吗？
- 如果会变，那么就说明这一部分代码属于核心逻辑，就应该放在train中
- 如果不会变，那么说明属于辅助功能，放入callback中



在完成上述代码的编写以后，我们即可编写`conf/config.yaml`等所有文件



## 6. train.py
`train.py` 负责实例化`hydra`与`solver`

## Appendix
for test：
```
CUDA_VISIBLE_DEVICES=7 python train.py train.epochs=3 dataset.max_samples=500 logger.name="test_run_01" callback.training_monitor.params.log_every_n_steps=50 dataset.batch_size=32
```