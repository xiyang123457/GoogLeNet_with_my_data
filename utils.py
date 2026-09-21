"""训练脚本共用工具：随机种子控制。

为什么单独成文件：model_train.py 与 train_transfer.py 需要同一套种子控制逻辑，
放在这里可以避免两份实现漂移 —— 改了一处忘了另一处，是复现性问题的常见来源。
（本仓库「已知限制」第 1 条：无全局随机种子，本文件是它的修复。）

用法：
    from utils import set_seed, seed_worker

    set_seed(42)                      # 在 __main__ 开头调用一次
    DataLoader(..., worker_init_fn=seed_worker, generator=g)
"""

import random

import numpy as np
import torch


def set_seed(seed=42):
    """固定全局随机种子，保证「同一份代码 + 同一个 seed」得到同一条曲线。

    训练过程中的随机性有 4 个来源，缺任何一个都无法复现：
      ① random / numpy   ：数据增强的随机裁剪、水平翻转
      ② torch            ：模型参数的随机初始化
      ③ torch.cuda/cuDNN ：卷积算法的选择（不同算法的浮点累加顺序不同）
      ④ DataLoader 子进程 ：num_workers>0 时每个 worker 有独立随机状态，
                            必须在 worker_init_fn 里重新播种（见 seed_worker）

    参数：
      seed: int，默认 42。任意整数都行，关键是同一个 seed 对应同一条曲线。
        想跑多种子实验时传 42 / 43 / 44 即可。

    坑：cudnn.deterministic=True 会禁用部分非确定性卷积算法，速度略降。
        本项目规模小，这个代价可以接受；若改成 False 追求速度，
        重跑结果会有微小浮动，复现性随之丢失。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """把主进程的种子分发给 DataLoader 的每个 worker 子进程。

    签名：seed_worker(worker_id: int) -> None
      作用：就地修改当前 worker 进程的 random / numpy 随机状态
      关键参数：worker_id 由 DataLoader 自动传入（0, 1, 2, ...），本函数用不到

    为什么需要它：DataLoader 在 num_workers>0 时会 fork 出子进程，子进程不继承
    主进程的 torch 随机状态，数据增强的随机性会脱离 set_seed 的控制。

    坑：必须配合 DataLoader(generator=g, worker_init_fn=seed_worker) 一起用。
        只写 worker_init_fn 而不传 generator，torch.initial_seed() 每次运行都
        不同，等于没固定。
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
