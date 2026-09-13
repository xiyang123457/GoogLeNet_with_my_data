from torch import nn
from torchvision import models
from torchvision.models import GoogLeNet_Weights


def build_transfer_model(num_classes=2, unfreeze_last_n=3):
    """构建迁移学习用的 GoogLeNet：

    - 加载 ImageNet 预训练权重（aux_logits=False，forward 直接返回 logits）
    - 把分类头 fc 替换为 num_classes 类
    - 先冻结全部参数，再解冻「分类头 fc」+「features 中最后 N 个 Inception block」
      按类型筛选 Inception 模块，精确匹配 inception4e / 5a / 5b，避免把中间
      的 MaxPool（无参数）误算进「最后 N 个 children」。
    """
    # 只传 weights：加载权重时 torchvision 会临时用 aux_logits=True 构建，
    # 加载完再把 aux_logits 置回 False（因为我们没显式传 aux_logits），
    # 最终 forward 直接返回 logits 张量，无需处理辅助输出。
    model = models.googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)

    # 替换分类头（原权重是 1000 类，直接丢弃）
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # 1) 全冻
    for p in model.parameters():
        p.requires_grad = False

    # 2) 解冻分类头
    for p in model.fc.parameters():
        p.requires_grad = True

    # 3) 解冻最后 N 个 Inception block
    #    注意：本 torchvision 版本的 GoogLeNet 没有 features 子模块，
    #    各层（conv1 / inception4e / fc 等）直接挂在模型顶层，故遍历 named_children。
    inceptions = [m for _, m in model.named_children()
                  if type(m).__name__ == 'Inception']
    for blk in inceptions[-unfreeze_last_n:]:
        for p in blk.parameters():
            p.requires_grad = True

    # 重新初始化分类头权重（正态小扰动，避免沿用 1000 类的尺度）
    nn.init.normal_(model.fc.weight, 0, 0.01)
    nn.init.constant_(model.fc.bias, 0)

    return model


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    m = build_transfer_model()
    children = list(m.named_children())
    print("顶层模块总数:", len(children))
    print("最后 5 个顶层模块:", [n for n, _ in children[-5:]])
    inceptions = [n for n, c in children if type(c).__name__ == 'Inception']
    print("最后 3 个 Inception block:", inceptions[-3:])
    print("可训练参数量:", count_trainable_params(m))
