import torch
import torch.nn as nn
import torch.fft as fft

class Linear(nn.Linear):
    r""" Linear layer for complex number inputs.
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__(in_features, out_features, False, device, dtype)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!

    def forward(self, x):
        x = torch.view_as_real(x).transpose(-2, -1)
        x = torch.nn.functional.linear(x, self.weight).transpose(-2, -1)
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.contiguous())
        return x
    
class CirculantAttention(nn.Module):
    r""" Circulant Attention
    https://arxiv.org/abs/2512.21542
    """
    def __init__(self, dim, proj_drop=0.):
        super().__init__()
        self.qkv = Linear(dim, dim * 3)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.SiLU())
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        h = w = int(n ** 0.5)

        # Prepare Q, K, V, T
        #    (1) qkv=fc(x), qkv=fft(qkv) is mathematically equivalent to x=fft(x), qkv=fc(x)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        #    (2) The latter requires fewer FFT computations, delivering higher throughput                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        t = self.gate(x)
        x = x.reshape(b, h, w, c)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        # Equation 15 of the paper
        #    (1) We use d=1 in practice, as discussed in Table 5
        #    (2) The 1/N factor is implicitly achieved by norm='ortho' in calculating Q, K
        attn = torch.conj(q) * k
        attn = torch.fft.irfft2(attn, s=(h, w), dim=(1, 2), norm='ortho')

        # Equation 16 of the paper
        attn = attn.reshape(b, n, c).softmax(dim=1).reshape(b, h, w, c)
        attn = torch.fft.rfft2(attn, dim=(1, 2))
        x = torch.conj(attn) * v
        x = torch.fft.irfft2(x, s=(h, w), dim=(1, 2), norm='ortho')

        # Output
        x = x.reshape(b, n, c) * t
        x = self.proj(x)
        return x


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(1, 1024, 32).to(device)  # B N C
    model = CirculantAttention(dim=32).to(device)

    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape) 
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")