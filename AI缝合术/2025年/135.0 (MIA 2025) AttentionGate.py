import pywt
# 装这个包 pip install pywavelets
import torch.nn as nn
import torch
import torch.nn.functional as F
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),                                                                                                                                 # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),                                                                                                                                 # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),                                                                                                                                 # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
      
# ------------张量测试---------------
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    g = torch.randn(1, 32, 256, 256, device=device)
    x = torch.randn(1, 32, 256, 256, device=device)
    att = AttentionGate(32, 32, 32).to(device)
    print(att)
    out = att(g, x)

    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")
    print("Input g:", g.shape)
    print("Input x:", x.shape)
    print("Output :", out.shape)
