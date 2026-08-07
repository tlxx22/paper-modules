import torch
import torch.nn as nn
import torch.nn.functional as F

# 论文题目：Dual-domain strip attention for image restoration
# 中文题目：双域条带注意力用于图像恢复
# 论文链接：https://doi.org/10.1016/j.neunet.2023.12.003
# 官方github：https://github.com/c-yn/DSANet
# 所属机构：
# 德国慕尼黑工业大学计算、信息与技术学院
# 代码整理：微信公众号：AI缝合术

# Frequency Strip Attention (FSA)
class FrequencyStripAttention(nn.Module):
    def __init__(self, k, kernel=7) -> None:
        super().__init__()

        self.channel = k

        self.vert_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.vert_high = nn.Parameter(torch.zeros(k, 1, 1))

        self.hori_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.hori_high = nn.Parameter(torch.zeros(k, 1, 1))

        self.vert_pool = nn.AvgPool2d(kernel_size=(kernel, 1), stride=1)
        self.hori_pool = nn.AvgPool2d(kernel_size=(1, kernel), stride=1)

        pad_size = kernel // 2
        self.pad_vert = nn.ReflectionPad2d((0, 0, pad_size, pad_size))
        self.pad_hori = nn.ReflectionPad2d((pad_size, pad_size, 0, 0))

        self.gamma = nn.Parameter(torch.zeros(k,1,1))
        self.beta = nn.Parameter(torch.ones(k,1,1))

    def forward(self, x):
        hori_l = self.hori_pool(self.pad_hori(x))
        hori_h = x - hori_l

        hori_out = self.hori_low * hori_l + (self.hori_high + 1.) * hori_h

        vert_l = self.vert_pool(self.pad_vert(hori_out))
        vert_h = hori_out - vert_l

        vert_out = self.vert_low * vert_l + (self.vert_high + 1.) * vert_h

        return x * self.beta + vert_out * self.gamma
    
if __name__ == "__main__":

    # 模块参数
    batch_size = 1    # 批大小
    channels = 32     # 输入特征通道数
    height = 256      # 图像高度
    width = 256        # 图像宽度
    
    # 创建 FSA 模块
    fsa = FrequencyStripAttention(k=channels, kernel=7)
    print(fsa)
    print("微信公众号:AI缝合术, nb!")
    
    # 生成随机输入张量 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)
    
    # 打印输入张量的形状
    print("Input shape:", x.shape)
    
    # 前向传播计算输出
    output = fsa(x)
    
    # 打印输出张量的形状
    print("Output shape:", output.shape)
