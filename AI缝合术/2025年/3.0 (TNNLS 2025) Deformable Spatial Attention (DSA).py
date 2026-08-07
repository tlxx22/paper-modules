import torch
import torch.nn as nn
from ops_dscn.modules import DSCNX,DSCNY

# 论文题目：DSAN: Exploring the Relationship between Deformable Convolution and Spatial Attention
# 中文题目：DSAN：探索可变形卷积与空间注意力之间的关系
# 论文链接：
# https://d197for5662m48.cloudfront.net/documents/publicationstatus/204101/preprint_pdf/8fd08fb8f6c83b6c5ad1eec37e87faf0.pdf
# 官方github：https://github.com/MarcYugo/DSAN-Deformable-Spatial-Attention
# 所属机构：温州大学等
# 代码整理：微信公众号：AI缝合术

class DSCNPair(nn.Module):
    def __init__(self, d_model, kernel_size, dw_kernel_size, pad, stride, dilation, group):
        super().__init__()
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.group = group
        self.conv0 = nn.Conv2d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        
        self.dscn_x = DSCNX(d_model, kernel_size, dw_kernel_size, stride=stride, pad=pad, dilation=dilation, group=group)#, offset_scale=0.4)
        self.dscn_y = DSCNY(d_model, kernel_size, dw_kernel_size, stride=stride, pad=pad, dilation=dilation, group=group)#, offset_scale=0.4)
        self.conv = nn.Conv2d(d_model, d_model, 1)

    def forward(self,x):
        u = x.clone()
        x = self.conv0(x)
        attn = x.permute(0,2,3,1)
        attn = self.dscn_x(attn,x)
        attn = self.dscn_y(attn,x)
        attn = attn.permute(0,3,1,2)
        attn = self.conv(attn)
        return u*attn
    
# Deformable Spatial Attention (DSA)
class DSA(nn.Module):
    def __init__(self, d_model, kernel_size, dw_kernel_size, pad, stride, dilation, group):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = DSCNPair(d_model, kernel_size, dw_kernel_size, pad, stride, dilation, group)
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
    # 将模块移动到 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建测试输入张量
    x = torch.randn(1, 64, 128, 128).to(device)
    # 初始化 dsa 模块
    dsa = DSA(d_model=64, kernel_size=11, dw_kernel_size=5, pad=5, stride=1, dilation=1, group=1)
    print(dsa)
    print("微信公众号:AI缝合术")
    dsa = dsa.to(device)
    # 前向传播
    output = dsa(x)
    
    # 打印输入和输出张量的形状
    print("输入张量形状:", x.shape)
    print("输出张量形状:", output.shape)