"""图像数据集逐通道均值/标准差计算工具（可复用）。

功能：读取图片文件夹（支持按类别子目录嵌套，递归扫描 jpg/png/bmp/webp 等常见格式），
对每张图计算逐通道像素均值，再基于该组逐通道均值做 IQR 四分位距异常检测：
某张图任意通道均值落在区间 [Q1 - k*IQR, Q3 + k*IQR] 之外，则整张图被剔除。
随后用剔除异常后的全部像素累加计算逐通道均值与标准差（总体标准差 ddof=0），
可直接作为训练时 transforms.Normalize 的 mean / std 参数。

说明
----
- 异常单位 = 整张图的逐通道均值（而非单个像素）。单像素 IQR 在天然连续的图像
  像素分布上几乎无区分度，而「某张图整体偏亮/偏暗/损坏」才是真正异常，按图为单位
  剔除更符合归一化数据清洗目的（鲁棒、无需假设分布）。
- 图像先 Resize(image_size)（默认 224，匹配 GoogLeNet 的 Resize((224,224))）再 /255
  转为 [0,1]，等价于 transforms.ToTensor()，保证算出的 mean/std 可直接喂给 Normalize。
  image_size=None 时按原图尺寸计算。
- 仅依赖 numpy + Pillow，不依赖 torch/torchvision，保持轻量独立。
- 既是命令行工具，也可 from compute_mean_std import compute_dataset_mean_std 复用。

示例（命令行）：
    python compute_mean_std.py --data_dir ./data
    python compute_mean_std.py --data_dir ./data --iqr_k 1.5 --image_size 224
"""

import argparse
import os

import numpy as np
from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif")
GRAY_MODES = ("L", "LA", "1", "I", "F", "P")


def _list_images(image_dir: str, recursive: bool) -> list[str]:
    """返回目录下（可递归）所有图片文件的绝对/相对路径，已排序保证可复现。"""
    paths: list[str] = []
    if recursive:
        for root, _, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith(IMAGE_EXTS):
                    paths.append(os.path.join(root, f))
    else:
        for f in os.listdir(image_dir):
            if f.lower().endswith(IMAGE_EXTS):
                paths.append(os.path.join(image_dir, f))
    return sorted(paths)


def _load_image_array(path: str, image_size, channels: int) -> np.ndarray:
    """读取单张图并转为 [0,1] 的 float32 数组，形状 (H, W, C)。"""
    with Image.open(path) as img:
        img = img.convert("RGB" if channels == 3 else "L")
        if image_size is not None:
            img = img.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:          # 灰度图补出通道维
        arr = arr[..., None]
    return arr


def compute_dataset_mean_std(
    image_dir: str,
    iqr_k: float = 1.5,
    image_size: int | None = 224,
    recursive: bool = True,
    force_channels: int | None = None,
) -> dict:
    """计算图像数据集逐通道像素均值与标准差，IQR 剔除异常图片。

    参数
    ----
    image_dir : str
        图片根目录（可含类别子目录）。
    iqr_k : float
        IQR 倍数，默认 1.5；判定区间为 [Q1 - k*IQR, Q3 + k*IQR]。
    image_size : int | None
        Resize 边长（与训练管线一致），默认 224；None / 0 表示用原图尺寸。
    recursive : bool
        是否递归子目录，默认 True。
    force_channels : int | None
        强制通道数：1=灰度，3=RGB；None=按首张有效图自动判定。

    返回
    ----
    dict
        {
          "mean": [per_channel], "std": [per_channel],
          "num_images", "num_valid", "num_excluded": int,
          "excluded": [paths], "errored": [paths], "channels": int,
        }
    """
    if image_size == 0:
        image_size = None
    image_dir = os.path.abspath(image_dir)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"图片目录不存在: {image_dir}")

    paths = _list_images(image_dir, recursive)
    if not paths:
        raise FileNotFoundError(f"在 {image_dir} 下未找到任何图片文件。")

    # ---- 确定通道数（按首张有效图自动判定，或强制） ----
    if force_channels in (1, 3):
        channels = force_channels
    else:
        channels = 3
        for p in paths:
            try:
                with Image.open(p) as tmp:
                    mode = tmp.mode
                channels = 1 if mode in GRAY_MODES else 3
                break
            except Exception:
                continue

    # ---- 第一遍：逐张读取，记录「逐图均值」并增量累加全局 sum/sumsq/count ----
    per_image_means: list[np.ndarray] = []
    per_image_sums: list[np.ndarray] = []
    per_image_sumsqs: list[np.ndarray] = []
    per_image_counts: list[int] = []
    errored: list[str] = []
    valid_paths: list[str] = []

    for p in paths:
        try:
            arr = _load_image_array(p, image_size, channels)
        except Exception:
            errored.append(p)
            continue
        n_pixels = arr.shape[0] * arr.shape[1]
        flat = arr.reshape(-1, channels)
        sums = flat.sum(axis=0).astype(np.float64)
        sumsqs = (flat ** 2).sum(axis=0).astype(np.float64)
        per_image_means.append(sums / n_pixels)
        per_image_sums.append(sums)
        per_image_sumsqs.append(sumsqs)
        per_image_counts.append(n_pixels)
        valid_paths.append(p)

    if not valid_paths:
        raise RuntimeError("所有图片均解码失败，无法计算统计量。")

    means_mat = np.stack(per_image_means)   # (N, C)
    num_valid = len(valid_paths)

    # ---- IQR 异常检测：任一通道均值越界则整图剔除 ----
    excluded_idx: set[int] = set()
    for c in range(channels):
        col = means_mat[:, c]
        q1 = np.percentile(col, 25)
        q3 = np.percentile(col, 75)
        iqr = q3 - q1
        lower = q1 - iqr_k * iqr
        upper = q3 + iqr_k * iqr
        for i in range(num_valid):
            if means_mat[i, c] < lower or means_mat[i, c] > upper:
                excluded_idx.add(i)

    excluded = [valid_paths[i] for i in sorted(excluded_idx)]

    # ---- 聚合「剔除异常后」的全局统计量（增量累加，内存友好） ----
    keep = [i for i in range(num_valid) if i not in excluded_idx]
    total_count = sum(per_image_counts[i] for i in keep)
    if total_count == 0:
        print("[警告] 所有图片都被判定为异常，无法计算统计量。")
        return {
            "mean": [0.0] * channels, "std": [0.0] * channels,
            "num_images": len(paths), "num_valid": num_valid,
            "num_excluded": len(excluded), "excluded": excluded,
            "errored": errored, "channels": channels,
        }

    total_sum = np.zeros(channels, dtype=np.float64)
    total_sumsq = np.zeros(channels, dtype=np.float64)
    for i in keep:
        total_sum += per_image_sums[i]
        total_sumsq += per_image_sumsqs[i]

    mean = total_sum / total_count
    var = total_sumsq / total_count - mean ** 2
    var = np.clip(var, 0.0, None)        # 浮点误差兜底
    std = np.sqrt(var)

    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "num_images": len(paths),
        "num_valid": num_valid,
        "num_excluded": len(excluded),
        "excluded": excluded,
        "errored": errored,
        "channels": channels,
    }


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="计算图像数据集逐通道均值/标准差（IQR 剔除异常图）。"
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        help="图片根目录，默认脚本同级 data 文件夹。",
    )
    parser.add_argument("--iqr_k", type=float, default=1.5, help="IQR 倍数，默认 1.5。")
    parser.add_argument(
        "--image_size", type=int, default=224,
        help="预处理 Resize 边长（与训练管线一致），默认 224；0 表示用原图尺寸。",
    )
    parser.add_argument(
        "--force_channels", type=int, default=None,
        help="强制通道数：1=灰度，3=RGB；默认按首图自动。",
    )
    parser.add_argument(
        "--no_recursive", action="store_true",
        help="不递归子目录，只扫描 data_dir 下一级。",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.force_channels not in (None, 1, 3):
        print("[错误] --force_channels 只能为 1 或 3。")
        return 1

    image_size = None if args.image_size in (0, None) else args.image_size
    try:
        res = compute_dataset_mean_std(
            args.data_dir,
            iqr_k=args.iqr_k,
            image_size=image_size,
            recursive=not args.no_recursive,
            force_channels=args.force_channels,
        )
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[错误] {e}")
        return 1

    print("\n==== 数据集均值/标准差统计 ====")
    print(f"图片目录    : {os.path.abspath(args.data_dir)}")
    print(f"通道数      : {res['channels']}")
    print(f"图片总数    : {res['num_images']}")
    print(f"成功解码    : {res['num_valid']}")
    print(f"异常剔除    : {res['num_excluded']}")
    if res["errored"]:
        print(f"解码失败    : {len(res['errored'])} 张（详见下方）")
    for c in range(res["channels"]):
        print(f"  通道 {c}: mean = {res['mean'][c]:.6f}   std = {res['std'][c]:.6f}")
    if res["excluded"]:
        print("\n被剔除的异常图片：")
        for p in res["excluded"]:
            print(f"  - {p}")
    if res["errored"]:
        print("\n解码失败的图片：")
        for p in res["errored"]:
            print(f"  - {p}")
    print("\nNormalize 可用参数：")
    print(f"  mean = {res['mean']}")
    print(f"  std  = {res['std']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
