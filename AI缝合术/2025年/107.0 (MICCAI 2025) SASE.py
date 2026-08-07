import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, act=True):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=(k - 1) // 2, groups=in_channels)                                                                                               # 微信公众号:AI缝合术
        if act:
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.dwconv(x))
        
class ConvBnoptinalAct(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,padding, with_act=True):
        super(ConvBnoptinalAct, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),                                                                                             # 微信公众号:AI缝合术
            nn.BatchNorm2d(out_channels)
            )
        self.with_act = with_act

    def forward(self, x):
        x = self.conv(x)
        if self.with_act == 'GELU':
            x = F.gelu(x)
        return x
        
class MultiSE(nn.Module):
    def __init__(self, in_channels, out_channels, if_deep_then_use=True, reduction=8, split=2):                                                                                             # 微信公众号:AI缝合术
        super(MultiSE, self).__init__()
        self.after_red_out_c = int(out_channels / reduction)
        self.add = (in_channels == out_channels)
        self.if_deep_then_use = if_deep_then_use

        if if_deep_then_use:
                # 使用 MultiSE 模块
            self.sigmoid = nn.Sigmoid()
            self.pwconv1 = ConvBnoptinalAct(in_channels, out_channels // reduction , kernel_size=1, padding=0)                                                                                               # 微信公众号:AI缝合术
            self.pwconv2 = ConvBnoptinalAct(out_channels // 2, out_channels, kernel_size=1, padding=0)
            self.m = nn.ModuleList(DWConv(self.after_red_out_c // split, self.after_red_out_c// split, k=3, act=False) for _ in range(reduction - 1))                                                                                               # 微信公众号:AI缝合术
        else:
                # 直接使用 DWConv + Pwconv
            self.dwconv = DWConv(in_channels, in_channels, k=3, act=True)
            self.pwconv = ConvBnoptinalAct(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x_residual = x
            # MultiSE 模块
        if self.if_deep_then_use:
            x = self.pwconv1(x)
                # import pdb;pdb.set_trace()
            x = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
            x.extend(m(x[-1]) for m in self.m)
            x[0] = x[0] + x[1]
            x.pop(1)

            y = torch.cat(x, dim=1)
            y = self.pwconv2(y)

        else:
            x = self.dwconv(x)
            y = self.pwconv(x)

        return x_residual + y if self.add else y 
    
if __name__ == "__main__":
    # 输入张量：形状为 (B, C, H, W)
    x = torch.randn(1, 32, 64, 64)

    # 初始化 MultiSE 模块（输入通道和输出通道相同，满足残差连接）
    multise = MultiSE(in_channels=32, out_channels=32, if_deep_then_use=True)                                                                                             # 微信公众号:AI缝合术

    # 前向传播测试
    output = multise(x)

    # 输出结果形状
    print(multise)
    print("\n微信公众号:AI缝合术\n")
    print("输入张量形状:", x.shape)       # [B, C, H, W]
    print("输出张量形状:", output.shape)  # [B, C, H, W]
