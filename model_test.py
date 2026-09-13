import torch
import torch.utils.data as Data
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST
from model import GoogLeNet, Inception


def test_data_process():
    test_data = FashionMNIST(root='./data',
                             train=False,
                             transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor(), ]),
                             download=True)

    test_dataloader = Data.DataLoader(dataset=test_data,
                                      batch_size=1,
                                      shuffle=True,
                                      num_workers=0)

    return test_dataloader


def test_model_process(model, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 只进行前向传播计算，把梯度置为0，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)

            model.eval()
            # 前向传播过程，输出预测值
            output = model(test_data_x)
            # 查找每一行最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            test_corrects += torch.sum(pre_lab == test_data_y)
            test_num += test_data_y.size(0)

    test_acc = test_corrects.double() / test_num
    print("测试的准确率:", test_acc)


if __name__ == '__main__':
    model = GoogLeNet(Inception)
    model.load_state_dict(torch.load('best_model.pth'))
    # 加载测试数据
    test_dataloader = test_data_process()

    # 加载模型测试的函数
    test_model_process(model, test_dataloader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandals', 'Shirt', 'Sneaker', 'Bag',
               'Ankle boot']
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 设置模型为验证模式
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
            label = b_y.item()
            print("预测值：", classes[result], "--------", "真实值:", classes[label])
