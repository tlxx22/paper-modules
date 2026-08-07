import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPoolAttention(nn.Module):  # 基于SE注意力
    def __init__(self, in_planes, reduction=16):
        super(DualPoolAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // reduction, in_planes, bias=False),
            nn.Sigmoid()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_planes, in_planes // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // reduction, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y).view(b, c, 1, 1)

        z = self.max_pool(x).view(b, c)
        z = self.fc2(z).view(b, c, 1, 1)
        x1 = x * y.expand_as(x)
        x2 = x * z.expand_as(x)
        x_sum = x1 + x2 + x
        return x_sum

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建一个随机输入特征图 
    input_tensor = torch.randn(1, 32, 256, 256).to(device)    # (batch_size, channels, height, width)
    
    dpa = DualPoolAttention(32).to(device) 
    print(dpa)
    
    # 前向传播
    output_tensor = dpa(input_tensor)
    
    # 打印输入输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")
