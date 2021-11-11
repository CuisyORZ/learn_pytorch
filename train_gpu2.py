import torch.optim
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from model import *


# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度：{},测试数据集长度：{}".format(train_data_size, test_data_size))

# DataLoader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 创建网络模型
class Csy(nn.Module):
    def __init__(self):
        super(Csy, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)

        )

    def forward(self, x):
        x = self.model(x)
        return x


csy = Csy()
csy = csy.to(device)

# 创建loos function
loss_f = nn.CrossEntropyLoss()
loss_f = loss_f.to(device)

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(params=csy.parameters(), lr=learning_rate)

# 设置训练网络参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs_learn_train")

for i in range(epoch):
    print("------第{}轮训练开始-------".format(i + 1))
    # 训练步骤
    csy.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = csy(imgs)
        loss = loss_f(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}， Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("learn_train_loss", loss.item(), total_train_step)

    # 测试步骤
    csy.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = csy(imgs)
            loss = loss_f(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            # 求准确率
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的准确率： {}".format(total_accuracy / test_data_size))
    writer.add_scalar("learn_test_loss", total_test_loss, total_test_step)
    writer.add_scalar("learn_test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(csy, "csy_{}.pth".format(i))
    print("模型已保存")

writer.close()
