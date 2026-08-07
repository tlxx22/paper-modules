import torch.nn as nn
import torch
from pytorch_wavelets import DWTForward, DWTInverse

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        # 使用 DWTForward 计算离散小波变换
        return self.DWT(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        # 使用 DWTInverse 计算逆离散小波变换
        return self.IDWT(x)

class RHDWT(nn.Module):

    def __init__(self, in_channels, n=1):
        super(RHDWT, self).__init__()

        # 卷积操作, 用于后续的反向传递
        self.identety = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * n, kernel_size=3, stride=2, padding=1)
        # DWT (离散小波变换)
        self.DWT = DWTForward(J=1, wave='haar')
        # 编码部分
        self.dconv_encode = nn.Sequential(
            nn.Conv2d(in_channels*4, in_channels * n, 3, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        # IDWT (离散小波逆变换)
        self.IDWT = DWTInverse(wave='haar')

    def _transformer(self, DMT1_yl, DMT1_yh):
        list_tensor = []
        a = DMT1_yh[0]
        list_tensor.append(DMT1_yl)
        for i in range(3):
            list_tensor.append(a[:, :, i, :, :])
        return torch.cat(list_tensor, 1)
    
    def forward(self, x):
        input = x
        # 获取 DWT 输出 (低频和高频部分)
        DMT1_yl, DMT1_yh = self.DWT(x)
        # 变换操作, 将高频部分合并到一起
        DMT = self._transformer(DMT1_yl, DMT1_yh)
        # 编码操作
        x = self.dconv_encode(DMT)
        # 身份卷积操作, 残差连接
        res = self.identety(input)
        # 输出加上残差
        out = torch.add(x, res)
        return out

if __name__ == '__main__':
    # 参数设置
    batch_size = 1               # 批量大小
    in_channels = 32             # 输入通道数
    height, width = 256, 256     # 输入图像的高度和宽度

    # 创建随机输入张量, 形状为 (batch_size, in_channels, height, width)
    x = torch.randn(batch_size, in_channels, height, width)
    # 创建 RHDWT 模型, n 用来扩张通道, 默认为1, 可实现特征图下采样(尺寸缩小一半, 通道数不变)
    model = RHDWT(in_channels=in_channels, n=1)
    # 打印模型结构
    print(model)
    # 进行前向传播, 得到输出
    output = model(x)
    # 打印输入和输出的形状
    print(f"输入张量的形状: {x.shape}")
    print(f"输出张量的形状: {output.shape}")