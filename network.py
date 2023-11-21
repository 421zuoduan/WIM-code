import torch
from torch import nn

# 先写好resnet的block块
class Res_block(nn.Module):
    def __init__(self,in_num,out_num,stride):
        super(Res_block, self).__init__()
        self.cov1 = nn.Conv2d(in_num,out_num,(3,3),stride=stride,padding=1)    #(3,3)  padding=1 则图像大小不变，stride为几图像就缩小几倍，能极大减少参数
        self.bn1 = nn.BatchNorm2d(out_num)
        self.cov2 = nn.Conv2d(out_num,out_num,(3,3),padding=1)
        self.bn2 = nn.BatchNorm2d(out_num)

        self.extra = nn.Sequential(
                nn.Conv2d(in_num,out_num,(1,1),stride=stride),
                nn.BatchNorm2d(out_num)
            )   #使得输入前后的图像数据大小是一致的
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.relu(self.bn1(self.cov1(x)))
        out = self.relu(self.bn2(self.cov2(out)))

        out = self.extra(x) + out
        return out

class Res_net(nn.Module):
    def __init__(self,num_class):
        super(Res_net, self).__init__()
        self.init = nn.Sequential(
            nn.Conv2d(3,16,(3,3)),
            nn.BatchNorm2d(16)
        )   #预处理
        self.bn1 = Res_block(16,32,2)
        self.bn2 = Res_block(32,64,2)
        self.bn3 = Res_block(64,128,2)
        self.bn4 = Res_block(128,256,2)
        self.fl = nn.Flatten()
        self.linear1 = nn.Linear(8192,10)
        self.linear2 = nn.Linear(10,num_class)
        self.relu = nn.ReLU()
    def forward(self,x):
        out = self.relu(self.init(x))
        #print('inint:',out.shape)
        out = self.bn1(out)
        #print('bn1:', out.shape)
        out = self.bn2(out)
        #print('bn2:', out.shape)
        out = self.bn3(out)
        #print('bn3:', out.shape)
        out = self.fl(out)
        #print('flatten:', out.shape)
        out = self.relu(self.linear1(out))
        #print('linear1:', out.shape)
        out = self.relu(self.linear2(out))
        #print('linear2:', out.shape)
        return out


#测试
def main():
    x = torch.randn(2, 3, 64, 64)
    net = Res_net(2)
    out = net(x)
    print(out.shape)


if __name__ == '__main__':
    main()