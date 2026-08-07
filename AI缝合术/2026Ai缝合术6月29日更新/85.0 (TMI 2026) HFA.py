import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or channels * 2
        self.net = nn.Sequential(
            ConvBlock(channels, hidden_channels, 3, 2),
            ConvBlock(hidden_channels, hidden_channels, 3, 2),
            ConvBlock(hidden_channels, hidden_channels, 3, 2)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or channels * 2
        self.proj1 = ConvBlock(hidden_channels, hidden_channels, 3, 1)
        self.proj2 = ConvBlock(hidden_channels, hidden_channels // 2, 3, 1)
        self.proj3 = ConvBlock(hidden_channels // 2, channels, 3, 1)
        self.out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, out_size):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.proj1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.proj2(x)
        x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)
        x = self.proj3(x)
        x = self.out(x)
        return x


class HFAM(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or channels * 2
        self.encoder = Encoder(channels, hidden_channels)
        self.decoder1 = Decoder(channels, hidden_channels)
        self.decoder2 = Decoder(channels, hidden_channels)

    def forward(self, fb, fm):
        if fb.shape != fm.shape:
            raise ValueError("fb and fm must have the same shape")

        out_size = fb.shape[-2:]
        fz = fb + fm

        z = self.encoder(fz)

        v1 = self.decoder1(z, out_size)
        v2 = self.decoder2(z, out_size)

        attn = torch.stack([v1, v2], dim=1)
        attn = torch.softmax(attn, dim=1)

        a1 = attn[:, 0]
        a2 = attn[:, 1]

        fe = a1 * fb + a2 * fm
        return fe


if __name__ == "__main__":
    fb = torch.randn(2, 64, 256, 256)
    fm = torch.randn(2, 64, 256, 256)

    model = HFAM(channels=64)
    fe = model(fb, fm)
    print(model)
    print(fb.shape)
    print(fm.shape)
    print(fe.shape)
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")                                                                                                                                                                              # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
