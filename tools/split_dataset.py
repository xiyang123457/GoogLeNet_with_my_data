"""数据集划分工具（可复用）。

功能：将「按类别分文件夹」存放的图片数据，按给定比例做分层抽样，
重新组织为 ``data/{train,test}/<class>/`` 结构（默认 train 85% / test 15%）。

说明
----
- 之前是 train/val/test 三路；按需求把「训练集 + 验证集」合并为统一的
  ``train``，因此现在只输出 ``train`` 和 ``test`` 两路。
- 仅依赖 Python 标准库（os / shutil / random / argparse），可整体拷贝到任意项目。
- 分层抽样：每个类别各自按比例切分，保证各子集类别比例一致。
- 固定随机种子，结果完全可复现。
- 既可作为命令行工具直接运行，也可 ``from split_dataset import split_image_dataset`` 复用。

示例（命令行）：
    python split_dataset.py --data_dir ./data
    python split_dataset.py --data_dir ./data --ratios 0.85 0.15 --seed 42
    python split_dataset.py --data_dir ./data --copy        # 复制而非移动（保留原图）
"""

import argparse
import os
import random
import shutil


def split_image_dataset(
    data_dir: str,
    splits: list[tuple[str, float]] | None = None,
    seed: int = 42,
    copy_mode: bool = False,
) -> dict[str, dict[str, int]]:
    """把一个 ``data_dir`` 下按类别分文件夹的图片做分层抽样并重新组织。

    参数
    ----
    data_dir : str
        数据集根目录，其每个直接子文件夹视为一个类别（输出子集名除外）。
    splits : list[tuple[str, float]], 可选
        子集名称与比例，如 ``[("train", 0.85), ("test", 0.15)]``。
        比例之和应约等于 1；默认 train 0.85 / test 0.15（即原 70%+15% 合并）。
    seed : int
        随机种子，固定后每次运行结果一致。
    copy_mode : bool
        True 表示复制文件（保留原图），False 表示移动文件（默认，原地重组织）。

    返回
    ----
    dict
        形如 ``{"cat": {"train": 85, "test": 15}, ...}`` 的统计字典。
    """
    if splits is None:
        splits = [("train", 0.85), ("test", 0.15)]

    names = [name for name, _ in splits]
    ratios = [ratio for _, ratio in splits]

    # ---- 参数校验 ----
    if len(names) != len(set(names)):
        raise ValueError(f"子集名称不能重复: {names}")
    if any(r <= 0 for r in ratios):
        raise ValueError("每个比例必须大于 0。")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"各比例之和必须等于 1，当前为 {sum(ratios):.4f}")

    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    # 若已经是划分结果，直接退出，避免重复划分破坏数据
    existing_splits = [n for n in names if os.path.isdir(os.path.join(data_dir, n))]
    if existing_splits:
        raise FileExistsError(
            f"检测到 {data_dir} 下已存在划分目录 {existing_splits}，请先清理或指定其他 data 目录，避免重复划分。"
        )

    # ---- 扫描类别 ----
    classes = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d not in names
    )
    if not classes:
        raise FileNotFoundError(f"在 {data_dir} 下没有找到任何类别子文件夹。")

    rng = random.Random(seed)  # 独立 Random 实例，避免污染调用方全局随机状态
    action = shutil.copy2 if copy_mode else shutil.move
    action_verb = "复制" if copy_mode else "移动"

    stats: dict[str, dict[str, int]] = {}

    for cls in classes:
        class_dir = os.path.join(data_dir, cls)
        files = [
            f for f in os.listdir(class_dir)
            if os.path.isfile(os.path.join(class_dir, f))
        ]
        if not files:
            print(f"[跳过] 类别 '{cls}' 为空目录。")
            continue

        total = len(files)
        rng.shuffle(files)  # 每类独立打乱（同一 seed 下结果可复现）

        # 按各比例计算每子集数量：前 N-1 个四舍五入，最后一个用余数补齐，
        # 保证各子集数量之和恰等于该类总数
        counts = [round(total * r) for r in ratios[:-1]]
        counts.append(total - sum(counts))
        counts = [max(c, 0) for c in counts]  # 极端小样本时兜底

        # 连续切片分配到各子集
        subsets: dict[str, list[str]] = {}
        start = 0
        for name, count in zip(names, counts):
            subsets[name] = files[start:start + count]
            start += count

        cls_stat: dict[str, int] = {}
        for name, subset_files in subsets.items():
            dst_dir = os.path.join(data_dir, name, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for f in subset_files:
                src = os.path.join(class_dir, f)
                dst = os.path.join(dst_dir, f)
                action(src, dst)
            cls_stat[name] = len(subset_files)

        # 移动模式下清理原类别空目录
        if not copy_mode and not os.listdir(class_dir):
            os.rmdir(class_dir)

        stats[cls] = cls_stat
        detail = " / ".join(f"{n} {cls_stat[n]}" for n in names)
        print(f"[{action_verb}] 类别 '{cls}': 共 {total} 张 -> {detail}")

    # ---- 总览 ----
    overview = {n: sum(s.get(n, 0) for s in stats.values()) for n in names}
    grand_total = sum(overview.values()) or 1
    print("\n==== 划分完成 ====")
    print(f"类别数      : {len(stats)}")
    print(f"样本总数    : {grand_total}")
    for n in names:
        print(f"{n:<8}      : {overview[n]} ({overview[n] / grand_total:.1%})")
    print(f"输出目录    : {data_dir}")
    print(f"处理方式    : {'复制（保留原图）' if copy_mode else '移动（原地重组织）'}")
    print(f"随机种子    : {seed}")

    return stats


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按类别分文件夹的图片数据集分层抽样工具（train/test）。"
    )
    parser.add_argument(
        "--data_dir", default="./data",
        help="数据集根目录，默认 ./data（其每个子文件夹视为一个类别）。",
    )
    parser.add_argument(
        "--ratios", nargs=2, type=float, default=[0.85, 0.15],
        metavar=("TRAIN", "TEST"),
        help="训练集/测试集比例，默认 0.85 0.15（train 含原训练+验证）。",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子，固定后每次运行结果一致，默认 42。",
    )
    parser.add_argument(
        "--copy", action="store_true",
        help="复制文件而非移动，保留原始数据目录不变。",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        split_image_dataset(
            data_dir=args.data_dir,
            splits=[("train", args.ratios[0]), ("test", args.ratios[1])],
            seed=args.seed,
            copy_mode=args.copy,
        )
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        print(f"[错误] {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
