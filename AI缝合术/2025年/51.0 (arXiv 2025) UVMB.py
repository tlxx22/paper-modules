import torch 
import torch.nn as nn
from mamba_ssm import Mamba


class UVMB(nn.Module):
    def __init__(self,c=3,w=256,h=256):
        super().__init__()
        self.convb  = nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=16, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=16, out_channels=c, kernel_size=3, stride=1, padding=1)
                        )
        self.model1 = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=c, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

        self.model2 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=c, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

        self.model3 = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=w*h, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        self.smooth = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, stride=1, padding=1)
        self.ln = nn.LayerNorm(normalized_shape=c)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        b,c,w,h = x.shape
        x = self.convb(x) + x
        x = self.ln(x.reshape(b, -1, c))
        y = self.model1(x).permute(0, 2, 1)
        z = self.model3(y).permute(0, 2, 1)
        att = self.softmax(self.model2(x))
        result = att * z
        output = result.reshape(b, c, w, h)
        return self.smooth(output)


if __name__ == "__main__":
    # 设置输入张量大小
    batch_size = 1
    channels = 3  # 输入图像的通道数
    height, width = 64, 64

    # 创建输入张量
    input_tensor = torch.randn(batch_size, channels, height, width).cuda()  # 输入张量

    # 初始化 UVMB 模块
    uvmb = UVMB(c=channels, w=height, h=width).cuda()
    print(uvmb)
    print("\n微信公众号: AI缝合术!\n")

    # 前向传播测试
    output = uvmb(input_tensor)

    # 打印输入和输出的形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
