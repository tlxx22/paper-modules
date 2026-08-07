import torch
from torch import nn

# 代码整理:微信公众号:AI缝合术

class TBFE(nn.Module):
    def __init__(self, input_channels, reduction_N = 32):
        super(TBFE, self).__init__()
        self.point_wise = nn.Conv2d(input_channels,reduction_N,kernel_size=1,padding=0,bias=False)    
        self.depth_wise = nn.Sequential(nn.Conv2d(reduction_N, reduction_N, kernel_size=(3, 3),padding=1),nn.BatchNorm2d(reduction_N),nn.ReLU(),)

        self.conv3D = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1,1,3),padding=(0,0,1),stride=(1,1,1),bias=False)
        self.bn = nn.BatchNorm2d(reduction_N)
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x_1 = self.point_wise(x)  
        x_2 = self.depth_wise(x_1)       
        x_2=x_1+x_2
        
        #DSC
        x_3 = x_1.unsqueeze(1)
        x_3 = self.conv3D(x_3)
        x_3 = x_3.squeeze(1)
        x = torch.cat((x_2,x_3),dim=1)
        
        return x
    
    
if __name__ == "__main__":
    # 模块参数
    batch_size = 1    # 批大小
    channels = 32     # 输入特征通道数
    height = 256      # 图像高度
    width = 256        # 图像宽度

    model = TBFE(input_channels = channels)
    print(model)
    print("微信公众号:AI缝合术, nb!")

    # 生成随机输入张量 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)
    # 打印输入张量的形状
    print("Input shape:", x.shape)
    # 前向传播计算输出
    output = model(x)
    # 打印输出张量的形状
    print("Output shape:", output.shape)
