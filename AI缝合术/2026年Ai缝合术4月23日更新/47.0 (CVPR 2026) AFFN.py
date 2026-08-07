import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce

class AFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(AFFN, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.patch_size = 8
        self.dim = dim

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        
        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        # 自相关融合权重
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 控制频域融合强度
        self.beta = nn.Parameter(torch.tensor(0.5))   # 控制空间域融合强度

    def forward(self, x):
        x = self.project_in(x)

        x_patch = rearrange(
            x, 'b c (h ph) (w pw) -> b c h w ph pw',
            ph=self.patch_size, pw=self.patch_size
        )

        # FFT
        Xf = torch.fft.rfft2(x_patch.float())
        Xf = Xf * self.fft
        # 自相关功率谱
        power = Xf * torch.conj(Xf)          # |X|^2
        R = torch.fft.irfft2(power, s=(self.patch_size, self.patch_size))                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        # 融合（频域 + 空间域）
        Xf_new = Xf + self.alpha * power     # 频域增强周期结构
        x_patch_new = torch.fft.irfft2(Xf_new, s=(self.patch_size, self.patch_size))                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        x_patch_new = x_patch_new + self.beta * R  # 空间域增强

        # 重组
        x = rearrange(
            x_patch_new, 'b c h w ph pw -> b c (h ph) (w pw)',
            ph=self.patch_size, pw=self.patch_size
        )

        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 64, 32, 32).to(device)

    model = AFFN(dim=64, ffn_expansion_factor=4, bias=False).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    print(model)
    
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")