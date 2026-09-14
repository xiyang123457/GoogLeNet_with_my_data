# 数据集划分工具 (split_dataset.py)

将「按类别分文件夹」存放的图片数据，按给定比例做**分层抽样**，重新组织为
`data/{train,test}/<class>/` 结构（默认 **train 85% / test 15%**）。

> 注：早期版本是 train/val/test 三路划分；后经调整把「训练集 + 验证集」合并为统一的
> `train`，因此当前只输出 `train` 和 `test` 两路。

## 特性

- **仅依赖 Python 标准库**（`os` / `shutil` / `random` / `argparse`），无需 torch 等环境，可整体拷贝到任意项目。
- **分层抽样**：每个类别各自按比例切分，保证各子集类别比例一致。
- **可复现**：固定随机种子，相同输入得到完全相同的划分结果。
- **安全防护**：若目标目录已存在 `train` / `test` 子目录，会报错退出，避免重复划分破坏数据。
- **双模式**：支持「移动」（原地重组织，默认）与「复制」（保留原图）两种方式。
- **两种用法**：既可作为命令行工具直接运行，也可 `from split_dataset import split_image_dataset` 在代码中复用。

## 目录约定

输入目录结构（`data` 下每个直接子文件夹视为一个类别）：

```
data/
├── cats/            # 类别 1
│   ├── 001.jpg
│   └── ...
└── dogs/            # 类别 2
    ├── 001.jpg
    └── ...
```

划分后结构：

```
data/
├── train/
│   ├── cats/  ...
│   └── dogs/  ...
└── test/
    ├── cats/  ...
    └── dogs/  ...
```

## 命令行用法

```powershell
# 默认 85% / 15% 划分（原地移动文件）
python split_dataset.py --data_dir ./data

# 指定比例（例如 80% / 20%）
python split_dataset.py --data_dir ./data --ratios 0.8 0.2

# 复制而非移动，保留原始 data/cats、data/dogs 目录
python split_dataset.py --data_dir ./data --copy

# 固定随机种子（默认 42，复现同一划分）
python split_dataset.py --data_dir ./data --seed 42
```

### 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--data_dir` | `./data` | 数据集根目录，其每个子文件夹视为一个类别。 |
| `--ratios` | `0.85 0.15` | 训练集 / 测试集比例，两个值之和必须等于 1。 |
| `--seed` | `42` | 随机种子，固定后划分结果可复现。 |
| `--copy` | 关闭 | 加上此参数表示复制文件（保留原图）；默认移动文件（原地重组织）。 |

### 运行示例（针对 GoogLeNet 猫狗数据）

```powershell
python split_dataset.py --data_dir "D:\pytorch_test\GoogLeNet_with_my_data\data"
```

## 作为库复用

```python
from split_dataset import split_image_dataset

stats = split_image_dataset(
    data_dir="path/to/data",
    splits=[("train", 0.85), ("test", 0.15)],
    seed=42,
    copy_mode=False,   # True=复制保留原图，False=移动
)

# stats 形如 {"cat": {"train": 85, "test": 15}, "dog": {...}}
print(stats)
```

### `split_image_dataset` 参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `data_dir` | `str` | 必填 | 数据集根目录。 |
| `splits` | `list[tuple[str, float]]` | `[("train",0.85),("test",0.15)]` | 子集名称与比例，比例之和须等于 1。 |
| `seed` | `int` | `42` | 随机种子。 |
| `copy_mode` | `bool` | `False` | `True` 复制保留原图，`False` 移动。 |

返回值为统计字典：`{"<类别>": {"<子集>": 数量, ...}, ...}`。

## 常见问题

- **误报「已存在划分目录」？** 说明 `data` 下已经有 `train` / `test` 文件夹，请先清理或换一个 `data_dir`，脚本不会覆盖已有划分。
- **划分后原 `cats`、`dogs` 目录没了？** 默认是「移动」模式会原地重组织；若想保留原图，加 `--copy`。
- **需要三路 train/val/test？** 修改 `--ratios` 不支持三路（CLI 固定两路），可用库模式传入 `splits=[("train",0.7),("val",0.15),("test",0.15)]`。
