import torch
import torch.nn as nn
import torch.nn.functional as F


class DAConv(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.num_neighbors = kernel_size * kernel_size - 1
        self.weight = nn.Parameter(torch.zeros(channels, self.num_neighbors))

        mask = torch.ones(kernel_size * kernel_size, dtype=torch.bool)
        mask[kernel_size * kernel_size // 2] = False
        self.register_buffer("mask", mask)

    def forward(self, x):
        b, c, h, w = x.shape
        x_pad = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="reflect")
        patches = F.unfold(x_pad, kernel_size=self.kernel_size)
        patches = patches.view(b, c, self.kernel_size * self.kernel_size, h * w)
        neighbors = patches[:, :, self.mask, :]
        weight = torch.softmax(self.weight, dim=-1).view(1, c, self.num_neighbors, 1)
        surround = (neighbors * weight).sum(dim=2).view(b, c, h, w)
        out = x - surround
        return out


class PSPModule(nn.Module):
    def __init__(self, in_channels, out_channels, bins=(1, 2, 3, 7)):
        super().__init__()
        self.bins = bins
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.GELU()
            )
            for bin_size in bins
        ])

    def forward(self, x):
        h, w = x.shape[-2:]
        outputs = []
        for stage in self.stages:
            feat = stage(x)
            feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=True)
            outputs.append(feat)
        return torch.cat(outputs, dim=1)


class AttentionGenerator(nn.Module):
    def __init__(self, channels, psp_bins=(1, 2, 3, 7), reduction=4):
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.daconv = DAConv(channels, kernel_size=3)
        self.psp = PSPModule(channels, hidden_channels, bins=psp_bins)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels + hidden_channels * len(psp_bins), hidden_channels, kernel_size=3, padding=1),                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        da_feat = self.daconv(x)
        psp_feat = self.psp(x)
        logits = self.fuse(torch.cat([da_feat, psp_feat], dim=1))
        attn = torch.softmax(logits.flatten(2), dim=-1).view(b, 1, h, w)
        attn = attn * h * w
        return attn


class CoordinateMapper(nn.Module):
    def __init__(self, kernel_size=7, sigma=2.0):
        super().__init__()
        assert kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        radius = kernel_size // 2
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        self.register_buffer("gaussian_kernel", kernel.view(1, 1, kernel_size, kernel_size))                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    def forward(self, attn):
        b, _, h, w = attn.shape
        device = attn.device
        dtype = attn.dtype

        y = torch.linspace(0, 1, h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        x = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        kernel = self.gaussian_kernel.to(device=device, dtype=dtype)

        denom = F.conv2d(attn, kernel, padding=self.padding).clamp_min(1e-6)
        u = F.conv2d(attn * y, kernel, padding=self.padding) / denom
        v = F.conv2d(attn * x, kernel, padding=self.padding) / denom

        u = u.clamp(0, 1)
        v = v.clamp(0, 1)

        u[:, :, 0, :] = 0
        u[:, :, -1, :] = 1
        v[:, :, :, 0] = 0
        v[:, :, :, -1] = 1

        grid = torch.cat([v * 2 - 1, u * 2 - 1], dim=1)
        grid = grid.permute(0, 2, 3, 1).contiguous()

        return grid, u, v


class ARS(nn.Module):
    def __init__(self, channels, psp_bins=(1, 2, 3, 7), coord_kernel_size=7, coord_sigma=2.0):
        super().__init__()
        self.attention_generator = AttentionGenerator(channels, psp_bins=psp_bins)
        self.coordinate_mapper = CoordinateMapper(kernel_size=coord_kernel_size, sigma=coord_sigma)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    def forward(self, x):
        attn = self.attention_generator(x)
        grid, u, v = self.coordinate_mapper(attn)
        out = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)

        return out


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 64, 128, 128).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = ARS(channels=64,psp_bins=(1, 2, 3, 7),coord_kernel_size=7,coord_sigma=2.0).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")