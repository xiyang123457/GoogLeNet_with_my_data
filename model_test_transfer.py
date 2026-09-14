import os
import torch
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)
from model_transfer import build_transfer_model

# ============================================================
# 迁移学习版测试脚本
# 测试对象：model_transfer.build_transfer_model 训练的权重 best_model_transfer.pth
# （torchvision 官方 GoogLeNet 预训练 + 冻结主干 + 微调 fc + 最后 3 个 Inception）
# 对应的训练脚本：train_transfer.py
# ============================================================

# 路径约定与训练脚本一致（以本文件所在目录为基准）
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# 迁移学习训练得到的权重（不要和手写版 best_model.pth 混淆）
BEST_MODEL_PATH = os.path.join(PROJECT_DIR, 'best_model_transfer.pth')
# 自己二分类数据的【独立测试集】目录（ImageFolder 结构：每个类别一个子文件夹）。
# 默认用 data/test（与 data/train 互不重叠，评估结果才可信）；
# 若没有独立测试集，再把它改回 data/train 做整体评估即可。
TEST_ROOT = os.path.join(PROJECT_DIR, 'data', 'test')

# 与训练一致的预处理：ImageNet 归一化常数（模型内部 transform_input 配合），
# 绝不能用手写版那套 [0.481,0.447,0.407]，否则分布错配精度崩。
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
test_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),  # 兼容灰度图，保证 3 通道
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])


def test_data_process(test_root=TEST_ROOT):
    test_data = ImageFolder(root=test_root, transform=test_transform)
    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=0)
    return test_dataloader, test_data.classes


def _set_chinese_font():
    """自动寻找一个支持中文的字体，避免图中中文乱码。"""
    import matplotlib.font_manager as fm
    candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Noto Sans CJK SC',
                  'WenQuanYi Micro Hei', 'Arial Unicode MS']
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams['font.sans-serif'] = [name]
            plt.rcParams['axes.unicode_minus'] = False
            return
    # Windows 常见字体文件兜底
    for path in [r'C:\Windows\Fonts\msyh.ttc',
                 r'C:\Windows\Fonts\simhei.ttf',
                 r'C:\Windows\Fonts\simsun.ttc']:
        if os.path.exists(path):
            prop = fm.FontProperties(fname=path)
            plt.rcParams['font.family'] = prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            return


def test_model_process(model, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()  # 关闭 dropout、BN 用 running stats

    classes = test_dataloader.dataset.classes
    # 二分类评估约定：正类 = 图片里有猫（cats），用于 ROC/PR 与混淆矩阵
    pos_class = 'cats'
    pos_idx = classes.index(pos_class)

    y_true = []   # 真实标签
    y_pred = []   # 预测标签
    y_score = []  # 正类预测概率

    test_corrects = 0
    test_num = 0

    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            output = model(test_data_x)  # logits

            probs = torch.softmax(output, dim=1)
            pre_lab = torch.argmax(output, dim=1)

            y_true.extend(test_data_y.cpu().numpy().tolist())
            y_pred.extend(pre_lab.cpu().numpy().tolist())
            y_score.extend(probs[:, pos_idx].cpu().numpy().tolist())

            test_corrects += int(torch.sum(pre_lab == test_data_y).item())
            test_num += test_data_y.size(0)

    test_acc = test_corrects / test_num

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    # 分类报告（Precision / Recall / F1 / Support）
    target_names = []
    for i, c in enumerate(classes):
        role = '正' if i == pos_idx else '负'
        target_names.append('{}({})'.format(c, role))
    print("\n测试的整体准确率:", round(test_acc, 4))
    print("\n分类报告（正类 = {}）：".format(pos_class))
    print(classification_report(y_true, y_pred,
                                target_names=target_names, digits=4))

    # 混淆矩阵（行 = 真实标签，列 = 预测标签）
    cm_raw = confusion_matrix(y_true, y_pred)
    if cm_raw.size != 4:
        cm_raw = np.zeros((2, 2), dtype=int)
    tn, fp, fn, tp = cm_raw.ravel()
    print("混淆矩阵（行 = 真实标签，列 = 预测标签）：")
    print("                 预测为负(0)   预测为正(1)")
    print("真实负(0) {:>10} {:>10}".format(int(tn), int(fp)))
    print("真实正(1) {:>10} {:>10}".format(int(fn), int(tp)))
    print("\nTN = {}  （真负：真没猫，判没猫）".format(int(tn)))
    print("FP = {}  （假正：真没猫，判有猫 / 误报）".format(int(fp)))
    print("FN = {}  （假负：真有猫，判没猫 / 漏检）".format(int(fn)))
    print("TP = {}  （真正：真有猫，判有猫）".format(int(tp)))

    return test_acc, y_true, y_pred, y_score, classes


def plot_test_results(test_acc, y_true, y_pred, y_score, classes, save_path,
                      title='Model Test'):
    _set_chinese_font()  # 先设置中文字体，避免图里中文乱码

    # 二分类约定：正类 = cats（图片里有猫）
    pos_class = 'cats'
    pos_idx = classes.index(pos_class)
    neg_idx = 1 - pos_idx

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 子图1：二分类混淆矩阵（按 负类/正类 重排，与示例一致）
    ax = axes[0]
    cm_raw = confusion_matrix(y_true, y_pred)
    if cm_raw.size != 4:
        cm_raw = np.zeros((2, 2), dtype=int)
    # 重排为 [[TN, FP], [FN, TP]]，行/列顺序 = [负类, 正类]
    cm = np.array([[cm_raw[neg_idx, neg_idx], cm_raw[neg_idx, pos_idx]],
                   [cm_raw[pos_idx, neg_idx], cm_raw[pos_idx, pos_idx]]])
    tn, fp, fn, tp = cm.ravel()

    im = ax.imshow(cm, cmap='Blues')
    ax.set_title('二分类混淆矩阵\n（正类 = 图片里有猫 / {}）'.format(pos_class))
    ax.set_xlabel('列 = 模型预测')
    ax.set_ylabel('行 = 真实标签')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['预测为负(0)\n{}'.format(classes[neg_idx]),
                        '预测为正(1)\n{}'.format(classes[pos_idx])])
    ax.set_yticklabels(['真实负(0)\n{}'.format(classes[neg_idx]),
                        '真实正(1)\n{}'.format(classes[pos_idx])])

    labels = [
        ['TN 真负\n真没猫，判没猫\n预测正确\n{}'.format(int(tn)),
         'FP 假正\n真没猫，判有猫\n误报（错）\n{}'.format(int(fp))],
        ['FN 假负\n真有猫，判没猫\n漏检（错）\n{}'.format(int(fn)),
         'TP 真正\n真有猫，判有猫\n预测正确\n{}'.format(int(tp))]
    ]
    text_colors = [['green', 'red'], ['red', 'green']]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j], ha='center', va='center',
                    color=text_colors[i][j], fontsize=10, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 子图2：ROC 曲线
    ax = axes[1]
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_idx)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label='ROC 曲线（AUC = {:.4f}）'.format(roc_auc))
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--',
            label='随机猜测')
    ax.set_title('ROC 曲线')
    ax.set_xlabel('FPR 假阳率')
    ax.set_ylabel('TPR 召回率')
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # 子图3：PR 曲线
    ax = axes[2]
    precision, recall, _ = precision_recall_curve(y_true, y_score,
                                                   pos_label=pos_idx)
    ap = average_precision_score(y_true, y_score, pos_label=pos_idx)
    pos_ratio = np.mean(y_true == pos_idx)
    ax.plot(recall, precision, color='teal', lw=2,
            label='PR 曲线（AP = {:.4f}）'.format(ap))
    ax.axhline(pos_ratio, color='gray', linestyle='--', lw=1.5,
               label='正类占比基线 = {:.2f}'.format(pos_ratio))
    ax.set_title('PR 曲线')
    ax.set_xlabel('Recall 召回率')
    ax.set_ylabel('Precision 查准率')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    fig.suptitle('{}  |  Overall Acc = {:.4f}'.format(title, test_acc),
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    print("测试可视化已保存至:", save_path)
    # 默认直接弹窗显示（图片已先由 savefig 落盘，关窗不影响文件）
    try:
        plt.show()
    except Exception as e:
        print("注意：弹窗显示失败（可能为无 GUI 环境），图片已保存至磁盘。", e)
    plt.close(fig)


if __name__ == '__main__':
    test_dataloader, classes = test_data_process()

    # 构建与训练时完全一致的结构（fc 输出 2 类），再载入迁移学习权重
    model = build_transfer_model()
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location='cpu'))

    # 一次性完成：整体准确率、分类报告、混淆矩阵、逐样本 ROC/PR 数据
    test_acc, y_true, y_pred, y_score, classes = \
        test_model_process(model, test_dataloader)

    save_path = os.path.join(PROJECT_DIR, 'result', 'model_transfer_test.png')
    plot_test_results(test_acc, y_true, y_pred, y_score, classes, save_path,
                      title='Transfer Learning Test')
