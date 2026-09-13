import copy
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
from torch import nn
from torch.optim import AdamW
from model_transfer import build_transfer_model

# =====================================================================
# 路径策略：以本代码文件所在目录为基准（相对定位，不写死盘符）
# =====================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_TRAIN = os.path.join(PROJECT_DIR, 'data', 'train')
MODEL_DIR = PROJECT_DIR
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_model_transfer.pth')  # 区别于手写版 best_model.pth

# 迁移学习关键：必须用 ImageNet 归一化常数（预训练权重基于此分布训练）
# 切勿沿用手写版自己算的 [0.481,0.447,0.407] 那套，否则输入分布错配会崩精度
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# 数据增强（迁移学习 + 小数据集，强度弱于手写版：去掉 ColorJitter，仅留裁剪+翻转）
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
# 验证集：纯净变换，不做增强（防止增强泄漏到验证集）
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])


def train_val_data_process():
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


def train_model_process(model, train_dataloader, val_dataloader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 分层学习率：分类头 fc 用较大 lr，解冻的 features 层用较小 lr（保护预训练特征）
    fc_params, feat_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (fc_params if name.startswith('fc') else feat_params).append(p)
    optimizer = AdamW([
        {'params': fc_params, 'lr': 1e-3},
        {'params': feat_params, 'lr': 1e-4},
    ], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # 验证集指标停滞时自动降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                           factor=0.5, patience=5)

    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0
    early_stop_patience = 8
    epochs_no_improve = 0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []

    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        train_loss = 0.0
        train_corrects = 0.0
        val_loss = 0.0
        val_corrects = 0.0
        train_num = 0
        val_num = 0

        # ---- 训练阶段 ----
        # 冻结层保持 eval（BN 用预训练 running stats，不被新数据污染），
        # 仅解冻的 fc + 分类头 dropout + 最后 3 个 Inception block 进入 train 模式。
        # 注：本 torchvision 版本 GoogLeNet 无 features 子模块，层直接挂在顶层。
        model.eval()
        model.fc.train()
        model.dropout.train()  # 分类头前的 dropout 正则生效
        unfrozen = [m for _, m in model.named_children()
                    if type(m).__name__ == 'Inception'][-3:]
        for blk in unfrozen:
            blk.train()

        for b_x, b_y in train_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y.data)
            train_num += b_x.size(0)

        # ---- 验证阶段：整体 eval ----
        model.eval()
        with torch.no_grad():
            for b_x, b_y in val_dataloader:
                b_x = b_x.to(device)
                b_y = b_y.to(device)
                output = model(b_x)
                pre_lab = torch.argmax(output, dim=1)
                loss = criterion(output, b_y)
                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        val_loss_all.append(val_loss / val_num)

        print('{} Train Loss: {:.4f} Train Acc: {:.4f}'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))

        scheduler.step(val_acc_all[-1])

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping: 连续 {} 轮验证准确率无提升，停止训练。".format(early_stop_patience))
                break

        time_use = time.time() - since
        print("训练耗费的时间{:.0f}m {:.0f}s".format(time_use // 60, time_use % 60))

    # 加载最高准确率下的模型参数再保存，避免保存次优的最后一轮权重
    model.load_state_dict(best_model_wts)
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(best_model_wts, BEST_MODEL_PATH)
    print('最优模型已保存到: {}'.format(BEST_MODEL_PATH))

    train_process = pd.DataFrame(data={"epoch": range(len(train_loss_all)),
                                       "train_loss_all": train_loss_all,
                                       "val_loss_all": val_loss_all,
                                       "train_acc_all": train_acc_all,
                                       "val_acc_all": val_acc_all})

    return train_process


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
    model = build_transfer_model()  # 默认解冻 fc + 最后 3 个 Inception block
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(model, train_dataloader, val_dataloader, num_epochs=30)
    matplot_acc_loss(train_process)
