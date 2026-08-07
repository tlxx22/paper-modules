import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmcv.runner import BaseModule   ### mmcv2.0更新以后的版本删除了runner等类
from mmengine.model import BaseModule

class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity

        identity = x

        x = self.projector(x)

        return identity + x

class Mona(BaseModule):
    def __init__(self,
                 in_dim,
                 factor=4):
        super().__init__()

        self.project1 = nn.Linear(in_dim, 64)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(64, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = MonaOp(64)

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim))

    def forward(self, x, hw_shapes=None):
        identity = x

        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2

if __name__ == "__main__":
    in_dim = 32
    batch_size = 1
    height, width = 224, 224 
    image = torch.randn(batch_size, in_dim, height, width).cuda()  # 输入张量

    # 创建输入张量
    input_tensor = torch.randn(batch_size, height * width, in_dim).cuda()  # 输入张量

    # 输入图像的尺寸
    hw_shapes = (height, width)

    # 初始化 Mona 模块
    mona = Mona(in_dim=in_dim).cuda()

    # 打印模型
    print(mona)
    print("\n微信公众号: AI缝合术!\n")

    # 前向传播测试
    output = mona(input_tensor, hw_shapes)

    # 打印输入和输出的形状
    print(f"Image shape: {image.shape}")
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
