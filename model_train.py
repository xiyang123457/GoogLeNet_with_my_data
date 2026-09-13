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
import torch.nn as nn

# =====================================================================
# 路径策略：所有路径都以“本代码文件所在目录”为基准（相对定位，
# 代码里不出现任何写死的盘符，如 d:/...）。
# 好处：
#   1. 无论在哪个目录下启动脚本，数据、模型都会落在本项目文件夹内；
#   2. 整个 LeNet 文件夹可直接复制改名成 AlexNet 等新项目复用，
#      无需修改任何路径。
# =====================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))  # 当前项目文件夹，即 GoogLeNet_with_my_data
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

    train_dataloader = Data.DataLoader(dataset=Data.Subset(train_ds, train_idx),
                                       batch_size=128,
                                       shuffle=True,
                                       num_workers=2)

    val_dataloader = Data.DataLoader(dataset=Data.Subset(val_ds, val_idx),
                                     batch_size=128,
                                     shuffle=False,
                                     num_workers=2)

    return train_dataloader, val_dataloader


# 模型训练
def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)  # 调试时务必看一眼

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)  # 第一轮：weight_decay 从 1e-4 降到 1e-5
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑，降低对训练样本过度自信
    # 验证集指标停滞时自动降低学习率，帮助跳出平台
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.5, patience=5)

    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高准确度
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

        # 学习率调度：根据验证准确率调整
        scheduler.step(val_acc_all[-1])

        # 保存最高准确度
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping: 连续 {} 轮验证准确率无提升，停止训练。".format(early_stop_patience))
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
    print('最优模型已保存到: {}'.format(BEST_MODEL_PATH))

    train_process = pd.DataFrame(data={"epoch": range(len(train_loss_all)),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all})

    return train_process


# 画图
def matplot_acc_loss(train_process):
    plt.figure(figsize=(12, 4))
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
    plt.show()


if __name__ == "__main__":
    GoogLeNet = GoogLeNet(Inception)
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(GoogLeNet, train_dataloader, val_dataloader, num_epochs=50)
    matplot_acc_loss(train_process)
