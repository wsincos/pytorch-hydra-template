"""
注册器提供了字符串名称到对象的映射。
它允许用户通过配置文件（其中包含字符串）来实例化特定的对象
（例如特定的骨干网络、求解器或数据加载器），而无需修改代码。

配置文件(YAML)只能存字符串，不能存 Python 类。
Registry 就是那个把字符串 'Transformer' 变成 Python 类 class Transformer 的翻译官


Registry包含如下内容:
    1. __init__: 初始化注册器，创建一个空的对象映射字典`_obj_map`。(`_obj_map`后续存储的就是具体的模型Class)
    2. register: 装饰器方法，用于注册类或函数到注册器中。
    3. get: 根据名称检索已注册的类或函数。


工作原理：
    1. 维护一个内部字典 `_obj_map`。
    2. 使用 `@register` 装饰器在模块导入时自动将类存入字典。
    3. 使用 `get` 方法根据字符串名称检索类。


使用示例：
    # 1. 定义并注册
    @MODEL_REGISTRY.register("MyModel")
    class MyModel(nn.Module): ...

    # 2. 配置文件 (config.yaml)
    model:
      name: "MyModel"

    # 3. 实例化 (builder.py)
    model_cls = MODEL_REGISTRY.get(cfg.model.name) # 这里的model_cls就是model对应的那个类
    model = model_cls(**cfg.params)
"""

class Registry:
    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def registrer(self, name=None):
        def _register(obj):
            key = name if name else obj.__name__
            if key in self._obj_map:
                raise KeyError(f"An object named '{key}' is already registered in '{self._name}' registry.")
            self._obj_map[key] = obj
            return obj
        return _register

    def get(self, name):
        if name not in self._obj_map:
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry.")
        return self._obj_map[name]

# 凡是需要在配置文件里随时替换的东西，都应该有 Registry

# === 1. 模型相关 ===
MODEL_REGISTRY = Registry("MODEL")         # 顶层Wrapper
ENCODER_REGISTRY = Registry("ENCODER")     # 编码器
DECODER_REGISTRY = Registry("DECODER")     # 解码器

# === 2. 数据相关 ===
DATASET_REGISTRY = Registry("DATASET")     # 数据集

# === 3. 训练相关 ===
OPTIMIZER_REGISTRY = Registry("OPTIMIZER") # 优化器 [Adam, SGD...]
SCHEDULER_REGISTRY = Registry("SCHEDULER") # 学习率调度器
CRITERION_REGISTRY = Registry("CRITERION") # 损失函数
CALLBACK_REGISTRY = Registry("CALLBACK")   # 回调函数