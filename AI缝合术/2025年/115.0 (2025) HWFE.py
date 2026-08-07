import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

# AI缝合术原创代码                                                                                                 
# --------- 1. DWT 和 IDWT -----------                                                                                                     # 微信公众号:AI缝合术
def dwt(x):
    """
    使用 PyWavelets 实现二维离散小波变换
    输入: x -> [B, C, H, W]
    输出: LL, LH, HL, HH -> [B, C, H//2, W//2]                                                                                                     # 微信公众号:AI缝合术
    """
    B, C, H, W = x.shape
    LL, LH, HL, HH = [], [], [], []
    for b in range(B):
        ll_, lh_, hl_, hh_ = [], [], [], []
        for c in range(C):
            coeffs2 = pywt.dwt2(x[b, c].cpu().numpy(), 'haar')                                                                                                     # 微信公众号:AI缝合术
            ll, (lh, hl, hh) = coeffs2
            ll_.append(torch.tensor(ll))
            lh_.append(torch.tensor(lh))
            hl_.append(torch.tensor(hl))
            hh_.append(torch.tensor(hh))
        LL.append(torch.stack(ll_))
        LH.append(torch.stack(lh_))
        HL.append(torch.stack(hl_))
        HH.append(torch.stack(hh_))
    return (torch.stack(LL).to(x.device),
            torch.stack(LH).to(x.device),
            torch.stack(HL).to(x.device),
            torch.stack(HH).to(x.device))

def idwt(ll, lh, hl, hh):
    """
    使用 PyWavelets 实现二维离散小波逆变换
    输入: LL, LH, HL, HH -> [B, C, H, W]
    输出: 重建图像 [B, C, H*2, W*2]
    """
    B, C, H, W = ll.shape
    out = []
    for b in range(B):
        rec = []
        for c in range(C):
            coeffs2 = (
                ll[b, c].detach().cpu().numpy(),
                (
                    lh[b, c].detach().cpu().numpy(),                                                                                                     # 微信公众号:AI缝合术
                    hl[b, c].detach().cpu().numpy(),                                                                                                     # 微信公众号:AI缝合术
                    hh[b, c].detach().cpu().numpy()                                                                                                     # 微信公众号:AI缝合术
                )
            )
            rec_ = pywt.idwt2(coeffs2, 'haar')                                                                                                     # 微信公众号:AI缝合术
            rec.append(torch.tensor(rec_))
        out.append(torch.stack(rec))
    return torch.stack(out).to(ll.device)

# --------- 2. Basic Block ----------
class BasicBlock(nn.Module):
    def __init__(self, channels):
        super(BasicBlock, self).__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)                                                                                                     # 微信公众号:AI缝合术
        self.ln = nn.GroupNorm(1, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)                                                                                                     # 微信公众号:AI缝合术
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)                                                                                                     # 微信公众号:AI缝合术

    def forward(self, x):
        residual = x
        out = self.dwconv(x)
        out = self.ln(out)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        return out + residual

# --------- 3. HWFE 模块 ----------
class HWFE(nn.Module):
    def __init__(self, channels):
        super(HWFE, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1),                                                                                                     # 微信公众号:AI缝合术
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.basic_block = BasicBlock(channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels * 4, kernel_size=1),                                                                                                     # 微信公众号:AI缝合术
            nn.BatchNorm2d(channels * 4),
            nn.GELU()
        )

    def forward(self, x):
        # Step 1: DWT分解为四个子带
        ll, lh, hl, hh = dwt(x)

        # Step 2: 通道拼接
        x = torch.cat([ll, lh, hl, hh], dim=1)

        # Step 3: 1×1 Conv + BN + GELU
        x = self.conv1(x)

        # Step 4: Basic Block
        x = self.basic_block(x)

        # Step 5: 1×1 Conv + BN + GELU
        x = self.conv2(x)

        # Step 6: 还原为四个子带
        B, C4, H, W = x.shape
        C = C4 // 4
        ll, lh, hl, hh = x[:, :C], x[:, C:2*C], x[:, 2*C:3*C], x[:, 3*C:]                                                                                                     # 微信公众号:AI缝合术

        # Step 7: IDWT还原图像
        out = idwt(ll, lh, hl, hh)
        return out

if __name__ == "__main__":

    # 输入张量：形状为 (B, C, H, W)
    x = torch.randn(1, 32, 256, 256)

    # 初始化 HWFE
    hwfe = HWFE(32)  

    # 前向传播测试
    output = hwfe(x)

    # 输出结果形状
    print(hwfe)
    print("\n微信公众号:AI缝合术\n")
    print("输入张量形状:", x.shape)       # [B, C, H, W]                                                                                             # 微信公众号:AI缝合术
    print("输出张量形状:", output.shape)  # [B, C, H, W]         
