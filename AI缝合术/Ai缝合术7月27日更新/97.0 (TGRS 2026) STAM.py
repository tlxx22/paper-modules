import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class ScaleSEBranch(nn.Module):
    def __init__(self, channels, scale=1, reduction=4):
        super().__init__()
        self.scale = scale
        self.se = SEBlock(channels, reduction)

    def forward(self, x, output_size):
        if self.scale > 1:
            h, w = x.shape[-2:]
            target_size = (
                max(h // self.scale, 1),
                max(w // self.scale, 1),
            )
            x = F.interpolate(
                x,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        x = self.se(x)

        if x.shape[-2:] != output_size:
            x = F.interpolate(
                x,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )

        return x


class STAM(nn.Module):
    def __init__(
        self,
        channels,
        reduction=4,
        scales=(1, 2, 4, 8),
        residual=False,
    ):
        super().__init__()

        if channels % 4 != 0:
            raise ValueError(
                f"channels must be divisible by 4, got {channels}"                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            )

        if len(scales) != 4:
            raise ValueError("scales must contain four values")                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        branch_channels = channels // 4
        self.residual = residual

        self.branches = nn.ModuleList(
            [
                ScaleSEBranch(
                    branch_channels,
                    scale=scale,
                    reduction=reduction,
                )
                for scale in scales
            ]
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.GELU(),
        )

    def forward(self, x):
        identity = x
        output_size = x.shape[-2:]
        chunks = torch.chunk(x, 4, dim=1)

        features = [
            branch(chunk, output_size)
            for branch, chunk in zip(self.branches, chunks)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        ]

        attention = self.fusion(
            torch.cat(features, dim=1)
        )

        out = identity * attention

        if self.residual:
            out = identity + out

        return out

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 64, 128, 128).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = STAM(64, reduction=4, scales=(1, 2, 4, 8), residual=True).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")