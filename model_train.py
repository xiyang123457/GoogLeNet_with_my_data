import copy
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
from model import GoogLeNet, Inception
from utils import set_seed, seed_worker  # 共用工具：随机种子控制（见 utils.py 文件头说明）
import torch.nn as nn

# =====================================================================
# 路径策略：所有路径都以“本代码文件所在目录”为基准（相对定位，
# 代码里不出现任何写死的盘符，如 d:/...）。
# 好处：
#   1. 无论在哪个目录下启动脚本，数据、模型都会落在本项目文件夹内；
#   2. 整个 LeNet 文件夹可直接复制改名成 AlexNet 等新项目复用，
#      无需修改任何路径。
# =====================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前项目文件夹（即脚本所在目录）
DATA_ROOT = os.path.join(PROJECT_DIR, 'data')  # 数据集目录（下载/读取都在这里）
MODEL_DIR = PROJECT_DIR  # 最优模型保存目录 = 当前项目文件夹内
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')  # 最优模型文件


# 数据加载
def train_val_data_process():
    # 定义数据集的路径（以脚本所在目录为基准，避免启动目录不同导致找不到数据）
    ROOT_TRAIN = os.path.join(PROJECT_DIR, 'data', 'train')
    # 归一化常数来自 compute_mean_std.py 对本数据集全量（data/，含 train+test）的统计结果
    # （IQR 已剔除异常图）。如需随数据变化自动重算，可 import 该脚本动态计算。
    normalize = transforms.Normalize(mean=[0.481451, 0.447649, 0.407904],
                                     std=[0.256604, 0.247922, 0.250483])

    # 训练集：数据增强（增强作用于 PIL 图，放在 ToTensor/Normalize 之前）
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 第一轮：暂时关闭，观察是否为欠拟合元凶
        transforms.ToTensor(),
        normalize,
    ])
    # 验证集：纯净变换，不做增强（避免增强泄漏到验证集、扭曲评估分布）
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    # 同一份固定 seed 切分，保证 train/val 严格不重叠、可复现
    full = ImageFolder(ROOT_TRAIN)
    train_idx, val_idx = Data.random_split(
        range(len(full)),
        [int(0.8 * len(full)), len(full) - int(0.8 * len(full))],
        generator=torch.Generator().manual_seed(42),
    )
    train_ds = ImageFolder(ROOT_TRAIN, transform=train_transform)
    val_ds = ImageFolder(ROOT_TRAIN, transform=val_transform)

    # 变量 g：torch.Generator，专门给 DataLoader 的 shuffle 顺序和 worker 播种用。
    #   例：g.manual_seed(42) 之后，每个 epoch 的 shuffle 顺序完全一致
    # 为什么不复用上面 random_split 的那个 Generator：那是临时对象且状态已被
    #   消耗过；这里用独立的新 Generator，职责清晰、互不干扰。
    g = torch.Generator()
    g.manual_seed(42)

    train_dataloader = Data.DataLoader(dataset=Data.Subset(train_ds, train_idx),
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=2,
                                       worker_init_fn=seed_worker,
                                       generator=g)

    val_dataloader = Data.DataLoader(dataset=Data.Subset(val_ds, val_idx),
                                     batch_size=128,
                                     shuffle=False,
                                     num_workers=2,
                                     worker_init_fn=seed_worker,
                                     generator=g)

    return train_dataloader, val_dataloader


# 模型训练
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)  # 调试时务必看一眼

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)  # 第二轮：加 BN 后 lr 降到 0.001 更稳
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑，降低对训练样本过度自信
    # 为什么 mode 从 'max' 改成 'min'：调度与早停的信号从「验证准确率」改成
    # 「验证损失」。验证集只有 345 张，1 个样本 = 0.29% 的准确率分辨率，
    # 用 acc 做信号会被噪声牵着走；loss 是连续量，分辨率高得多。
    # （本仓库「已知限制」第 5 条的修复）
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=5)

    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 变量 best_loss：float，历史最低验证损失，作为选模与早停的依据。
    #   例：0.1823
    # 为什么用 float('inf') 而不是 0.0：loss 越小越好，用无穷大保证第 0 轮
    #   一定会刷新一次记录，不会因为初始值选错而永远不保存模型。
    best_loss = float('inf')
    # 变量 best_acc：float，只用于打印「最优轮对应的验证准确率」，不参与选模。
    best_acc = 0.0
    # 早停：连续 early_stop_patience 轮验证准确率无提升则停止
    early_stop_patience = 10
    epochs_no_improve = 0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 测试集准确度列表
    val_acc_all = []

    since = time.time()  # 时间

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        # 初始化
        train_loss = 0.0
        train_corrects = 0.0

        val_loss = 0.0
        val_corrects = 0.0
        # 样本数量
        train_num = 0
        val_num = 0
        # 对每一个minibatch进行计算
        # 设置模型为训练模式
        model.train()
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 输入为一个batch,输出为一个batch中的预测
            output = model(b_x)
            # 通过一个softmax转化为概率,取概率最大的一个值作为标签
            pre_lab = torch.argmax(output, dim=1)
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)
            # 将梯度初始化为0
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 根据反向传播的梯度信息来更新参数
            optimizer.step()
            # loss是一个样本的平均损失,乘以size得到一个minibatch的损失
            train_loss += loss.item() * b_x.size(0)
            # 预测正确则正确数加一，后续用正确数除以总数据得到正确率
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置模型为评估模式
            model.eval()
            with torch.no_grad():
                # 输入为一个batch,输出为一个batch中的预测
                output = model(b_x)
                # 通过一个softmax转化为概率,取概率最大的一个值作为标签
                pre_lab = torch.argmax(output, dim=1)
                # 计算每一个batch的损失函数
                loss = criterion(output, b_y)
                # loss是一个样本的平均损失,乘以size得到一个minibatch的损失
                val_loss += loss.item() * b_x.size(0)
                # 预测正确则正确数加一，后续用正确数除以总数据得到正确率
                val_corrects += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)

        # 计算loss值和准确率
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)

        val_acc_all.append(val_corrects.double().item() / val_num)
        val_loss_all.append(val_loss / val_num)

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 学习率调度：根据验证损失调整（loss 越低越好 → mode='min'）
        scheduler.step(val_loss_all[-1])

        # 选模：以验证损失为准，同时记下该轮的验证准确率用于打印
        if val_loss_all[-1] < best_loss:
            best_loss = val_loss_all[-1]
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping: 连续 {} 轮验证损失无下降，停止训练。".format(early_stop_patience))
                break

        # 训练
        time_use = time.time() - since
        print("训练耗费的时间{:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    # 选择最优参数
    # 加载最高准确率下的模型参数
    model.load_state_dict(best_model_wts)
    # 保存最优模型（相对本代码文件所在目录，模型落在当前项目文件夹内）
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(best_model_wts, BEST_MODEL_PATH)
    print('最优模型（val loss = {:.4f}，对应 val acc = {:.4f}）已保存到: {}'.format(
        best_loss, best_acc, BEST_MODEL_PATH))

    train_process = pd.DataFrame(data={"epoch": range(len(train_loss_all)),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all})

    return train_process


# 画图
def matplot_acc_loss(train_process, save_path=None):
    # 方法签名 -> None
    #   作用：画「loss 双线 + acc 双线」两个子图，可选落盘
    #   关键参数：save_path —— str 或 None。传路径则先 savefig 落盘再弹窗；
    #             传 None 则只弹窗（保持原有行为）
    #   坑：plt.show() 在无 GUI 环境（服务器/容器）会阻塞或报错，
    #       此时设 MPLBACKEND=Agg 可跳过弹窗、只走 savefig 分支。
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process['epoch'], train_process.train_loss_all, 'ro-', label="Train Loss")
    plt.plot(train_process['epoch'], train_process.val_loss_all, 'bs-', label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, 'ro-', label="Train Acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, 'bs-', label="Val Acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Acc")

    # 为什么先 savefig 再 show：show() 是阻塞调用，用户关窗后才返回。
    # 先落盘可以保证「关了窗也拿到图」，也避免无 GUI 环境下 show 失败导致图丢失。
    # （本仓库「已知限制」第 7 条的修复：原来只有 plt.show()，曲线只能靠窗口截图）
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
        print('训练曲线已保存到: {}'.format(save_path))
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # 全局随机种子：覆盖模型初始化 + 数据增强 + DataLoader 顺序。
    # 想跑多种子实验时改这里的数字即可（例如 42 / 43 / 44）。
    set_seed(42)
    GoogLeNet = GoogLeNet(Inception)
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(GoogLeNet, train_dataloader, val_dataloader, num_epochs=50)
    # 新图用 curve_*.png 命名，与 result/ 下 5 张历史截图（窗口截图）区分开
    matplot_acc_loss(train_process,
                     save_path=os.path.join(PROJECT_DIR, 'result', 'curve_scratch.png'))
