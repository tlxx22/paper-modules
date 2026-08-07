import torch
import torch.nn as nn

class NeLU(nn.Module):
    def __init__(self, alpha: float = 0.2) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, (-self.alpha) / (1 + x**2))
    
    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, torch.tensor(1.0, device=x.device), self.alpha * (2 * x) / (1 + x**2)**2)

if __name__ == "__main__":

    # 创建输入张量：形状 [B, C, H, W]
    x = torch.randn(1, 3, 2, 2).cuda()

    # NeLU
    act = NeLU().cuda()
    output = act(x)
    
    print("\nNeLU: \n微信公众号:AI缝合术\n")

    # 打印输入输出数据和形状
    print("\n输入形状:", x.shape)     # [B, C, H, W]
    print('\ninput:',x)
    print("\n输出形状:", output.shape)  # [B, C_out, H_out, W_out]
    print('\noutput:',output)
