import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 拓展实验，搭建 Resnet
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel), )
        # [b,in_channel,h,w]=>[b,out_channel,h,w]
        self.extra = nn.Sequential()
        if out_channel != in_channel:
            self.extra = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channel), )

    def forward(self, x):
        out = self.block(x)
        out = out + self.extra(x)  # resnet 核心，将输入 x 与输出 out 相加
        out = nn.functional.relu(out, inplace=True)
        return out


class ResNet18(nn.Module):
    def __init__(self, item=10):
        super(ResNet18, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResBlock(64, 64, 1),
            ResBlock(64, 64, 1),
            ResBlock(64, 128, 2),
            ResBlock(128, 128, 1),
            ResBlock(128, 256, 2),
            ResBlock(256, 256, 1),
            ResBlock(256, 512, 2),
            ResBlock(512, 512, 1),
            # [b,512,h,w]=>[b,512,1,1]
            nn.AdaptiveAvgPool2d([1, 1])
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, item),
        )

    def forward(self, x):
        x = self.Conv(x)
        print(x.shape)
        x = self.linear(x)
        return x