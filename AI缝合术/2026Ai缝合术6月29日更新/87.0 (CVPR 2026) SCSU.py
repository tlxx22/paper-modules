import torch
import torch.nn as nn
import torch.nn.functional as F


class SCSU(nn.Module):
    def __init__(self, channels, k=3, bias=False):
        super().__init__()
        if k % 2 == 0:
            raise ValueError("k must be an odd integer")                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        strip_k = 3 * k + 2

        self.dw_square = nn.Conv2d(
            channels,
            channels,
            kernel_size=k,
            padding=k // 2,
            groups=channels,
            bias=bias
        )

        self.dw_hstrip = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, strip_k),
            padding=(0, strip_k // 2),
            groups=channels,
            bias=bias
        )

        self.dw_vstrip = nn.Conv2d(
            channels,
            channels,
            kernel_size=(strip_k, 1),
            padding=(strip_k // 2, 0),
            groups=channels,
            bias=bias
        )

        self.weight_gen = nn.Conv2d(
            channels,
            channels * 3,
            kernel_size=1,
            bias=True
        )

    def forward(self, x):
        b, c, h, w = x.shape

        w_select = F.adaptive_avg_pool2d(x, 1)
        w_select = self.weight_gen(w_select)
        w_select = w_select.view(b, 3, c, 1, 1)
        w_select = torch.softmax(w_select, dim=1)

        y1 = self.dw_square(x)
        y2 = self.dw_hstrip(x)
        y3 = self.dw_vstrip(x)

        y = torch.stack([y1, y2, y3], dim=1)
        y = (y * w_select).sum(dim=1)

        return y


class SpatialChannelSelection(nn.Module):
    def __init__(self, channels, k1=3, k2=5, bias=False):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("channels must be divisible by 2")                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        hidden = channels // 2

        self.scsu1 = SCSU(hidden, k=k1, bias=bias)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.scsu2 = SCSU(hidden, k=k2, bias=bias)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        self.proj = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1 = self.scsu1(x1)
        y2 = self.scsu2(x2)
        y = torch.cat([y1, y2], dim=1)
        y = self.proj(y)
        return y


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 64, 128, 128).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = SpatialChannelSelection(64, k1=3, k2=5, bias=False).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")