# data/ 目录说明

本目录中的数据文件未纳入版本控制（`.gitignore` 中排除了 `data/*`，仅保留本文件）。克隆仓库后本目录为空，需要自行准备数据。

## 数据来源

| 项目 | 内容 |
|---|---|
| 数据集 | Cat and Dog（Cats and Dogs dataset to train a DL model） |
| 来源 | https://www.kaggle.com/datasets/tongpython/cat-and-dog |
| 许可证 | CC0: Public Domain（Kaggle 页面标注） |
| 原始规模 | 约 10.0k 个文件，228.46 MB（Version 1） |
| 本项目使用 | 2025 张（训练 1721 张，测试 304 张） |

对外提交或发表时，请注明数据集名称、Kaggle 链接与访问日期。Kaggle 页面标注为 CC0，但该页面是上游猫狗数据集的再分发版本，用于正式发表前建议核对上游原始条款。

## 关于本项目所用子集

本项目只使用了上游数据集的一部分（2025 张，约占 20%）。**这批数据的筛选与去重过程没有留存记录**，当时为手工挑选，没有脚本、日志或随机种子。

因此有两点需要注意：

- 本目录中的数据无法精确重建，换一批图片重做会得到不同的准确率；
- 文中涉及数据规模的判断都需要谨慎对待，参见仓库根目录 README 的「已知限制」第 5 条。

## 目录结构

数据按 PyTorch `ImageFolder` 格式组织：

```
data/
├── train/
│   ├── cats/   *.jpg   （860 张）
│   └── dogs/   *.jpg   （861 张）
└── test/
    ├── cats/   *.jpg   （152 张）
    └── dogs/   *.jpg   （152 张）
```

## 准备数据

从 Kaggle 下载并解压。若目录结构为按类别分文件夹（如 `data/cats`、`data/dogs`），可用 `tools/split_dataset.py` 划分：

```bash
python tools/split_dataset.py --data_dir ./data            # 默认 85% / 15%，移动模式
python tools/split_dataset.py --data_dir ./data --copy     # 复制模式，保留原图
```

参数说明见 `tools/split_dataset_README.md`。

## 权重文件

`best_model.pth` 与 `best_model_transfer.pth` 同样未纳入版本控制，克隆后需要自行训练才能得到。

即使取得这两个文件，文中从零版的 82.24% 也无法还原：原 `best_model.pth` 已被后续重跑覆盖，用现存文件在 `data/test` 上重新计算得到 82.57%，且错误较多的类别由「猫」变为「狗」。参见仓库根目录 README 的「已知限制」第 6 条。
