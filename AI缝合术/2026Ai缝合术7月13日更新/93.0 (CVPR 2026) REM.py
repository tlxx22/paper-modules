import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        act=True,
        norm=True
    ):
        super().__init__()

        if padding is None:
            if isinstance(kernel_size, tuple):
                padding = tuple(k // 2 for k in kernel_size)
            else:
                padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=not norm
        )
        self.bn = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HaarDWT2D(nn.Module):
    def __init__(self):
        super().__init__()

        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) * 0.5
        lh = torch.tensor([[1.0, -1.0], [1.0, -1.0]]) * 0.5
        hl = torch.tensor([[1.0, 1.0], [-1.0, -1.0]]) * 0.5
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) * 0.5

        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filters", filters)

    def forward(self, x):
        b, c, h, w = x.shape

        pad_h = h % 2
        pad_w = w % 2

        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        weight = self.filters.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)

        y = F.conv2d(
            x,
            weight,
            stride=2,
            padding=0,
            groups=c
        )

        y = y.view(b, c, 4, y.shape[-2], y.shape[-1])

        ll = y[:, :, 0]
        lh = y[:, :, 1]
        hl = y[:, :, 2]
        hh = y[:, :, 3]

        return ll, lh, hl, hh, (h, w)


class HaarIDWT2D(nn.Module):
    def __init__(self):
        super().__init__()

        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) * 0.5
        lh = torch.tensor([[1.0, -1.0], [1.0, -1.0]]) * 0.5
        hl = torch.tensor([[1.0, 1.0], [-1.0, -1.0]]) * 0.5
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) * 0.5

        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("filters", filters)

    def forward(self, ll, lh, hl, hh, out_size=None):
        b, c, h, w = ll.shape

        x = torch.stack([ll, lh, hl, hh], dim=2)
        x = x.view(b, c * 4, h, w)

        weight = self.filters.to(device=x.device, dtype=x.dtype).repeat(c, 1, 1, 1)

        y = F.conv_transpose2d(
            x,
            weight,
            stride=2,
            padding=0,
            groups=c
        )

        if out_size is not None:
            out_h, out_w = out_size
            y = y[:, :, :out_h, :out_w]

        return y


class REM(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        hidden_channels=None,
        use_residual=True,
        norm=True
    ):
        super().__init__()

        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or out_channels                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        self.use_residual = use_residual

        self.in_proj = ConvBNAct(
            in_channels,
            hidden_channels,
            kernel_size=1,
            padding=0,
            norm=norm
        )

        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        self.ll_branch = ConvBNAct(
            hidden_channels,
            hidden_channels,
            kernel_size=1,
            padding=0,
            norm=norm
        )

        self.lh_branch = nn.Sequential(
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=(1, 3),
                norm=norm
            ),
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=(3, 1),
                norm=norm
            ),
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                norm=norm
            )
        )

        self.hl_branch = nn.Sequential(                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=(3, 1),
                norm=norm
            ),
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=(1, 3),
                norm=norm
            ),
            ConvBNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                norm=norm
            )
        )

        self.hh_branch = ConvBNAct(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            norm=norm
        )

        self.out_proj = ConvBNAct(
            hidden_channels,
            out_channels,
            kernel_size=1,
            padding=0,
            act=False,
            norm=norm
        )

        if in_channels == out_channels:
            self.shortcut = nn.Identity()                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        else:
            self.shortcut = ConvBNAct(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                act=False,
                norm=norm
            )

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.in_proj(x)

        ll, lh, hl, hh, out_size = self.dwt(x)

        ll = self.ll_branch(ll)
        lh = self.lh_branch(lh)
        hl = self.hl_branch(hl)
        hh = self.hh_branch(hh)

        x = self.idwt(ll, lh, hl, hh, out_size=out_size)
        x = self.out_proj(x)

        if self.use_residual:
            x = x + identity

        return x


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 64, 128, 128).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = REM(64, 64).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")