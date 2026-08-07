import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)

        return x
    


if __name__ == "__main__":
    batch_size = 1
    height, width = 256, 256  # 输入图像的大小
    dim = 32  # 输入通道数
    ffn_expansion_factor = 4  # 扩展因子
    bias = True  # 偏置

    # 创建输入张量
    input_tensor = torch.randn(batch_size, dim, height, width).cuda()  # 输入张量

    # 初始化 EDFFN 模块
    edffn = EDFFN(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias).cuda()
    # 打印模型
    print(edffn)
    print("\n微信公众号: AI缝合术!\n")

    # 前向传播
    output = edffn(input_tensor)

    # 打印输入和输出的形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
