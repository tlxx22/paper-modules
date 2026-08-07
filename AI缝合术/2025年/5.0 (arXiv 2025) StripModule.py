import torch
import torch.nn as nn

# 论文题目：Strip R-CNN: Large Strip Convolution for Remote Sensing Object Detection
# 中文题目：Strip R-CNN：用于遥感目标检测的大条带卷积
# 论文链接：https://arxiv.org/pdf/2501.03775?
# 官方github： https://github.com/HVision-NKU/Strip-R-CNN
# 所属机构：南开大学,湖南先进技术研发学院等
# 代码整理：微信公众号：AI缝合术

class StripConv(nn.Module):
    def __init__(self, dim, k1, k2):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial1 = nn.Conv2d(dim,dim,kernel_size=(k1, k2), stride=1, padding=(k1//2, k2//2), groups=dim)     
        self.conv_spatial2 = nn.Conv2d(dim,dim,kernel_size=(k2, k1), stride=1, padding=(k2//2, k1//2), groups=dim)

        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):   
        attn = self.conv0(x)
        attn = self.conv_spatial1(attn)
        attn = self.conv_spatial2(attn)
        attn = self.conv1(attn)

        return x * attn
    
class StripModule(nn.Module):
    def __init__(self, d_model,k1,k2):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = StripConv(d_model,k1,k2)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
        
if __name__ == "__main__":
    # 模块参数
    batch_size = 1    # 批大小
    channels = 32     # 输入特征通道数
    height = 256      # 图像高度
    width = 256        # 图像宽度
   
    sm = StripModule(d_model=channels, k1=1, k2=19)
    print(sm)
    print("微信公众号:AI缝合术")

    # 生成随机输入张量 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)

    # 打印输入张量的形状
    print("Input shape:", x.shape)

    # 前向传播计算输出
    output = sm(x)

    # 打印输出张量的形状
    print("Output shape:", output.shape)
