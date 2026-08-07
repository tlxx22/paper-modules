import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文题目：Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection
# 中文题目：Strip R-CNN：用于遥感目标检测的大条带卷积
# 论文链接：https://arxiv.org/pdf/2501.03775?
# 官方github： https://github.com/HVision-NKU/Strip-R-CNN
# 所属机构：南开大学,湖南先进技术研发学院等
# 代码整理：微信公众号：AI缝合术


class FirstOctaveConv(nn.Module):   # 对应第一个红色和绿色
      
    def __init__(self, in_channels, out_channels,kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(FirstOctaveConv, self).__init__()
        self.stride = stride
        kernel_size = kernel_size[0]   # 3
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * in_channels), # (512,256)
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels, in_channels - int(alpha * in_channels), 
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):    # x：n,c,h,w
        if self.stride ==2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x) # 低频
        X_h = x
        X_h = self.h2h(X_h)   # 高频
        X_l = self.h2l(X_h2l) # 低频

        return X_h, X_l

class OctaveConv(nn.Module): # 低、高频输入，低、高频输出 对应第二个红色和绿色
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        # 低到低，通道缩一半
        self.l2l = torch.nn.Conv2d(int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        
        # 低到高，改变输出通道
        self.l2h = torch.nn.Conv2d(int(alpha * in_channels), out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        
        # 高到低，输出通道减一半，改变输入通道
        self.h2l = torch.nn.Conv2d(in_channels - int(alpha * in_channels), int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        
        # 高到高，输出、输入通道都改变
        self.h2h = torch.nn.Conv2d(in_channels - int(alpha * in_channels),
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)

        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]),int(X_h2h.size()[3])), mode='bilinear')

        X_h = X_l2h + X_h2h
        X_l = X_h2l + X_l2l

        return X_h, X_l

class LastOctaveConv(nn.Module): # 低频和高频对齐输出
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(LastOctaveConv, self).__init__()   # 继承 nn.Module 的一些属性和方法
        self.stride = stride
        kernel_size = kernel_size[0]
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.l2h = torch.nn.Conv2d(int(alpha * out_channels), out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(out_channels - int(alpha * out_channels),
                                   out_channels,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        X_h, X_l = x

        if self.stride == 2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2h = self.h2h(X_h) # 高频组对齐通道
        X_l2h = self.l2h(X_l) # 低频组对齐通道
        # 低频组对齐长宽尺寸
        X_l2h = F.interpolate(X_l2h, (int(X_h2h.size()[2]), int(X_h2h.size()[3])), mode='bilinear') 

        X_h = X_h2h + X_l2h  
        return X_h       

# Frequency-aware module(FAM)
class Octave(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super(Octave, self).__init__()
        # 第一层，将特征分为高频和低频
        self.fir = FirstOctaveConv(in_channels, out_channels, kernel_size)

        # 第二层，低高频输入，低高频输出
        self.mid1 = OctaveConv(in_channels, in_channels, kernel_size)   # 同频输入、输出
        self.mid2 = OctaveConv(in_channels, out_channels, kernel_size)  # 不同频输入、输出

        # 第三层，将低高频汇合后输出
        self.lst = LastOctaveConv(in_channels, out_channels, kernel_size)

    def forward(self, x):   
        x0 = x
        x_h, x_l = self.fir(x)                   
        x_hh, x_ll = x_h, x_l,
        # x_1 = x_hh +x_ll
        x_h_1, x_l_1 = self.mid1((x_h, x_l))     
        x_h_2, x_l_2 = self.mid1((x_h_1, x_l_1)) 
        x_h_5, x_l_5 = self.mid2((x_h_2, x_l_2)) 
        x_ret = self.lst((x_h_5, x_l_5)) 
        return x_ret
        
if __name__ == "__main__":
    # 模块参数
    batch_size = 1    # 批大小
    channels = 32     # 输入特征通道数
    height = 256      # 图像高度
    width = 256        # 图像宽度
   
    fam = Octave(in_channels=32, out_channels=32, kernel_size=(3,3))
    print(fam)
    print("微信公众号:AI缝合术, nb!")

    # 生成随机输入张量 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)

    # 打印输入张量的形状
    print("Input shape:", x.shape)

    # 前向传播计算输出
    output = fam(x)

    # 打印输出张量的形状
    print("Output shape:", output.shape)
