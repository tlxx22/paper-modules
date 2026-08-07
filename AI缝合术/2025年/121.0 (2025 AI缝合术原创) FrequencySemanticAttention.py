import torch
from torch import nn

# AI缝合术原创代码            
class ChannelAttentionModule3D(nn.Module):
    def __init__(self, in_features, reduction_rate=16):
        super(ChannelAttentionModule3D, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features, in_features // reduction_rate),                                                                                                                      # 微信公众号:AI缝合术
            nn.ReLU(),
            nn.Linear(in_features // reduction_rate, in_features)                                                                                                                      # 微信公众号:AI缝合术
        )

    def forward(self, x):
        max_x = torch.max(x, dim=2, keepdim=True)[0]
        max_x = torch.max(max_x, dim=3, keepdim=True)[0]
        max_x = torch.max(max_x, dim=4, keepdim=True)[0]

        avg_x = torch.mean(x, dim=2, keepdim=True)
        avg_x = torch.mean(avg_x, dim=3, keepdim=True)
        avg_x = torch.mean(avg_x, dim=4, keepdim=True)

        max_x = self.linear(max_x.squeeze())
        avg_x = self.linear(avg_x.squeeze())

        att = max_x + avg_x
        att = torch.sigmoid(att).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)                                                                                                                      # 微信公众号:AI缝合术

        return x * att
    
class FrequencyChannelAttention3D(nn.Module):
    def __init__(self, num_channels, num_slices , uncertainty=True, rank=5):                                                                                                                      # 微信公众号:AI缝合术
        super(FrequencyChannelAttention3D, self).__init__()
        self.Channel_att = ChannelAttentionModule3D(num_channels)


    def forward(self, x):
        freq_domain = torch.fft.fftn(x, dim=(2, 3, 4), norm='ortho')
        freq_real = freq_domain.real
        freq_imag = freq_domain.imag

        Channel_attn = self.Channel_att(freq_real)
        freq_real = freq_real * Channel_attn
        freq_imag = freq_imag * Channel_attn

        freq_domain_combined = torch.complex(freq_real, freq_imag)
        x_modified = torch.fft.ifftn(freq_domain_combined, dim=(2, 3, 4), norm='ortho').real                                                                                                                      # 微信公众号:AI缝合术

        return x_modified

if __name__ == "__main__":

    # 输入张量：形状为 (B, C, D, H, W)
    x = torch.randn(1, 32, 8, 64, 64)  # 例如 batch=1, 通道=32, 深度=8, 高=64, 宽=64

    # 初始化
    fsa = FrequencyChannelAttention3D(num_channels=32, num_slices=8)

    # 前向传播测试
    output = fsa(x)

    # 输出结果形状
    print(fsa)
    print("\n微信公众号:AI缝合术\n")
    print("输入张量形状:", x.shape)      # [B, C, D, H, W]                                                                                                                      # 微信公众号:AI缝合术
    print("输出张量形状:", output.shape)  # [B, C, D, H, W]                                                                                                                      # 微信公众号:AI缝合术

