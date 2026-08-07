import torch
import torch.nn as nn

class BSiLU(nn.Module):
    def __init__(self, alpha: float = 1.67) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.alpha) * torch.sigmoid(x) - self.alpha / 2

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) + (x + self.alpha) * torch.sigmoid(x) * (1 - torch.sigmoid(x))

if __name__ == "__main__":

    # 创建输入张量：形状 [B, C, H, W]
    x = torch.randn(1, 3, 256, 256).cuda()

    # BSiLU
    act = BSiLU().cuda()
    output = act(x)
    
    print("\nBSiLU: \n微信公众号:AI缝合术\n")

    # 打印输入输出形状
    print("输入形状:", x.shape)     # [B, C, H, W]
    print("输出形状:", output.shape)  # [B, C_out, H_out, W_out]
