import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def ka_window_partition(x, window_size):
    """
    input: (B, H*W, C)
    output: (B, num_windows*C, window_size, window_size)
    """
    B, L, C = x.shape
    H, W = int(sqrt(L)), int(sqrt(L))
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(B, -1, window_size, window_size)
    return windows


def ka_window_reverse(windows, window_size, H, W):
    """
    input: (B, num_windows*C, window_size, window_size)
    output: (B, H*W, C)
    """
    B = windows.shape[0]
    x = windows.contiguous().view(B, H // window_size, W // window_size, -1, window_size, window_size)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous().view(B, H*W, -1)
    return x

class WinKernel_Reweight(nn.Module):
    def __init__(self, dim, win_num=4):
        super().__init__()
        
        self.dim = dim
        self.win_num = win_num
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.downchannel = nn.Conv2d(win_num*dim, win_num, kernel_size=1, groups=win_num)
        self.linear1 = nn.Conv2d(win_num, win_num*4, kernel_size=1)
        self.gelu = nn.GELU()
        self.linear2 = nn.Conv2d(win_num*4, win_num, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, kernels, windows):
        """
        kernels:  win_num*c, c, k ,k
        windows:  bs, win_num*c, wh, ww
        """

        B = windows.shape[0]

        # win_weight:  bs, win_num*c, 1, 1
        win_weight = self.pooling(windows)

        # win_weight:  bs, win_num, 1, 1
        win_weight = self.downchannel(win_weight)

        win_weight = self.linear1(win_weight)
        win_weight = win_weight.permute(0, 2, 3, 1).reshape(B, 1, -1)
        win_weight = self.gelu(win_weight)
        win_weight = win_weight.transpose(1, 2).reshape(B, -1, 1, 1)
        win_weight = self.linear2(win_weight)
        # weight:  bs, win_num, 1, 1, 1, 1
        weight = self.sigmoid(win_weight).unsqueeze(-1).unsqueeze(-1)

        # kernels:  1, win_num, c, c, k, k
        kernels = kernels.reshape(self.win_num, self.dim, self.dim, kernels.shape[-2], kernels.shape[-1]).unsqueeze(0)

        # kernels:  bs, win_num, c, c, k, k
        kernels = kernels.repeat(B, 1, 1, 1, 1, 1)

        # kernels:  bs, win_num, c, c, k, k
        kernels = weight * kernels

        # kernels:  bs*win_num*c, c, k, k
        kernels = kernels.reshape(-1, self.dim, kernels.shape[-2], kernels.shape[-1])

        return kernels
    


class WinReweight(nn.Module):
    """
    第一个分组卷积产生核，然后计算核的自注意力，调整核，第二个分组卷积产生输出，skip connection
    
    Args:
        dim: 输入通道数
        window_size: 窗口大小
        num_heads: 注意力头数
        qkv_bias: 是否使用偏置
        qk_scale: 缩放因子
        attn_drop: 注意力dropout
        proj_drop: 输出dropout
        ka_window_size: kernel attention window size
        kernel_size: 卷积核大小
        stride: 卷积步长
        padding: 卷积padding
    """

    def __init__(self, dim, input_resolution, num_heads, ka_win_num=4, kernel_size=3, stride=1, padding=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_num = ka_win_num
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        self.scale = qk_scale or (dim//num_heads) ** (-0.5)
        self.window_size = int(input_resolution // sqrt(ka_win_num))
        self.num_layers = self.win_num
        
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.downchannel = nn.Conv2d(dim*ka_win_num, ka_win_num, kernel_size=1, groups=ka_win_num)
        self.linear1 = nn.Conv2d(win_num, win_num*4, kernel_size=1)
        self.gelu = nn.GELU()
        self.linear2 = nn.Conv2d(win_num*4, win_num, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = self.input_resolution, self.input_resolution

        X = X.permute(0, 2, 3, 1).reshape(B, -1, C)

        # x_windows:  bs, win_num*c, wh, ww
        x_windows = ka_window_partition(x, self.window_size)

        # x_windows： bs, win_num*c, wh, ww
        x_windows = self.pooling(x_windows)

        # win_weight:  bs, win_num, 1, 1
        x_windows = self.downchannel(x_windows)

        x_windows = self.linear1(x_windows)
        x_windows = x_windows.permute(0, 2, 3, 1).reshape(B, 1, -1)
        x_windows = self.gelu(x_windows)
        x_windows = x_windows.transpose(1, 2).reshape(B, -1, 1, 1)
        x_windows = self.linear2(x_windows)
        # weight:  bs, win_num, 1, 1
        weight = self.sigmoid(x_windows)

        return weight









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
    def __init__(self, item=10, win_size=2):
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

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.reweight = WinReweight(dim=512, win_size=2)
        self.linear1 = nn.Linear(512, item)

    def forward(self, x):

        # b, 512, h, w
        x = self.Conv(x)

        weight = self.reweight(x)

        x_flatten = self.flatten(x)
        x_flatten = self.dropout(x_flatten)
        
        x = self.linear1(x_flatten)
        x = x * weight
        x = self.linear2(x)
        return x