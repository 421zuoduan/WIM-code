import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def dataset():
    #加载数据，并划分数据集和测试集大小
    train_data = torchvision.datasets.CIFAR10(root="F:\CIFAR-10-100(含png图)\dataset", train=True, download=True,
                                            transform=torchvision.transforms.Compose(
                                                [
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3801,))
                                                ]
                                            ))
    test_data = torchvision.datasets.CIFAR10(root="F:\CIFAR-10-100(含png图)\dataset", train=False, transform=torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3801,))
        ]
    ),
    download=True)
    # train_data=Subset(train_data,range(0,50000))#划分训练集，大小为10K
    # test_data = Subset(test_data,range(0,10000))#划分测试集，大小为10K
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print(f"训练集的长度为：{train_data_size}")
    print(f"测试集的长度为：{test_data_size}")

