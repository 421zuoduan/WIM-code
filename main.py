import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset, sampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F
import copy
from resnet18 import ResNet18
from network import Res_net
from dataset import dataset

NUM_TRAIN = 49000

# 数据预处理，减去cifar-10数据均值
transform_normal = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载训练集
cifar10_train = dset.CIFAR10('数据集/data/CIFAR10', train=True, download=True, transform=transform_normal)
train_loader = DataLoader(cifar10_train, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

# 加载验证集
cifar10_val = dset.CIFAR10('数据集/data/CIFAR10', train=True, download=True, transform=transform_normal)
val_loader = DataLoader(cifar10_val, batch_size=64, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

# 加载测试集
cifar10_test = dset.CIFAR10('数据集/data/CIFAR10', train=False, download=True, transform=transform_normal)
test_loader = DataLoader(cifar10_test, batch_size=64)

learning_rate = 1e-4
epochs = 50
model = ResNet18().to(device)
# model = Res_net(2).to(device)
# optimizer_resnet = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
# optimizer_resnet = optim.Adam(model.parameters(), lr=learning_rate)

losses = []
accuracies = []

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#网络训练&测试

# 记录训练次数
total_train_step = 0
# 初始化测试集最高正确率
best_accuracy = 0
# 训练的轮数
epoch = 30
#tensorboard
writer = SummaryWriter("logs")
for i in range(epoch):
    print(f"----------第{i + 1}轮训练开始了----------")
    # 训练开始
    total_train_right = 0  # 每一轮开始时将正确个数置0
    model.train()
    for datas in train_loader:
        data, targets = datas
        data = data.to(device)
        targets = targets.to(device)#将数据放在device上
        output = model(data)
        train_loss = loss_fn(output, targets)  # 计算网络的loss
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()  # 反向传播算法更新权重
        train_right = (output.argmax(1) == targets).sum().item()  # 与标签一致则预测正确
        total_train_right += train_right
        total_train_step += 1
        if total_train_step % 100 == 0:
            # 每100次输出一次loss
            print(f"训练次数：{total_train_step},Loss:{train_loss.item()}")
    train_accuracy = total_train_right / len(train_data)  # 计算训练集上的正确率
    print(f"训练集上的正确率:{train_accuracy}")
    # 测试开始
    cnn.eval()
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for datas in test_loader:
            data, targets = datas
            data = data.to(device)
            targets = targets.to(device)
            output = cnn(data)
            test_loss = loss_fn(output, targets)  # 测试集上的loss
            total_test_loss = total_test_loss + test_loss.item()
            test_right = (output.argmax(1) == targets).sum().item()
            total_test_accuracy = total_test_accuracy + test_right
    total_test_accuracy = total_test_accuracy / len(test_data)
    avarage_test_loss = total_test_loss / len(test_data)
    print(f"验证集上的平均Loss:{total_test_loss}")
    print(f"验证集上的正确率:{total_test_accuracy}")
    writer.add_scalars("loss", {
        "train_Loss": train_loss.item(), "test_loss": avarage_test_loss}, i)
    writer.add_scalars("accuracy", {
        "train_accuracy": train_accuracy, "test_accuracy": total_test_accuracy}, i)  # 使用tensorboard绘制loss曲线和accuracy曲线
    if total_test_accuracy > best_accuracy:
        best_epoch = total_train_step
        best_accuracy = total_test_accuracy  # 判断并更新最高正确率
        torch.save(cnn.state_dict(),
                   "model_best.pth")  # 将测试集表现最佳的模型权重保存起来
print("最高正确率：", best_accuracy, "best time: ", best_epoch)