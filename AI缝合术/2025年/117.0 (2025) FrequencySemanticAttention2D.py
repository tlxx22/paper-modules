import torch
from torch import nn

# AI缝合术原创代码（2D版本）
class SemanticAttentionModule2D(nn.Module):
    def __init__(self, in_features, reduction_rate=16):
        super(SemanticAttentionModule2D, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, in_features // reduction_rate),                                                                                                                      # 微信公众号:AI缝合术
            nn.ReLU(),
            nn.Linear(in_features // reduction_rate, in_features)                                                                                                                      # 微信公众号:AI缝合术
        )

    def forward(self, x):
        # x: (B, C, H, W)
        max_x = torch.max(x, dim=2, keepdim=True)[0]  # H 方向
        max_x = torch.max(max_x, dim=3, keepdim=True)[0]  # W 方向

        avg_x = torch.mean(x, dim=2, keepdim=True)
        avg_x = torch.mean(avg_x, dim=3, keepdim=True)

        max_x = self.linear(max_x.squeeze(-1).squeeze(-1))
        avg_x = self.linear(avg_x.squeeze(-1).squeeze(-1))

        att = max_x + avg_x
        att = torch.sigmoid(att).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)                                                                                                                      # 微信公众号:AI缝合术

        return x * att


class FrequencySemanticAttention2D(nn.Module):
    def __init__(self, num_channels, uncertainty=True, rank=5):
        super(FrequencySemanticAttention2D, self).__init__()
        self.semantic_att = SemanticAttentionModule2D(num_channels)                                                                                                                      # 微信公众号:AI缝合术

    def forward(self, x):
        # 频域变换 (H, W)
        freq_domain = torch.fft.fftn(x, dim=(2, 3), norm='ortho')                                                                                                                      # 微信公众号:AI缝合术
        freq_real = freq_domain.real
        freq_imag = freq_domain.imag

        # 语义注意力
        semantic_attn = self.semantic_att(freq_real)
        freq_real = freq_real * semantic_attn
        freq_imag = freq_imag * semantic_attn

        # 合成复数并反变换
        freq_domain_combined = torch.complex(freq_real, freq_imag)
        x_modified = torch.fft.ifftn(freq_domain_combined, dim=(2, 3), norm='ortho').real                                                                                                                      # 微信公众号:AI缝合术

        return x_modified


if __name__ == "__main__":
    # 输入张量：形状为 (B, C, H, W)
    x = torch.randn(1, 32, 64, 64)  # batch=1, 通道=32, 高=64, 宽=64

    # 初始化
    fsa = FrequencySemanticAttention2D(num_channels=32)

    # 前向传播测试
    output = fsa(x)

    # 输出结果形状
    print(fsa)
    print("\n微信公众号:AI缝合术\n")
    print("输入张量形状:", x.shape)      # [B, C, H, W]                                                                                                                       # 微信公众号:AI缝合术
    print("输出张量形状:", output.shape)  # [B, C, H, W]                                                                                                                       # 微信公众号:AI缝合术
