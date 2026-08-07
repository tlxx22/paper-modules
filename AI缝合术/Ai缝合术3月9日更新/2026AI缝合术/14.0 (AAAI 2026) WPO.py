import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_dct  # 该库需提前安装：pip install torch-dct

class WPO(nn.Module):
    """
    Wave Propagation Operator
    Wave equation operator:
    d2u/dt2 - c2(d2u/dx2 + d2u/dy2) + αdu/dt = 0;
    du/dx_{x=0, x=a} = 0
    du/dy_{y=0, y=b} = 0
    =>
    A_{n, m} = C(a, b, n==0, m==0) * sum_{0}^{a}{ sum_{0}^{b}{\phi(x, y)cos(n\pi/ax)cos(m\pi/by)dxdy }}
    core = cos(n\pi/ax)cos(m\pi/by) * (1 - [(n\pi/a)^2 + (m\pi/b)^2]c2t2) * e^(-αt)
    u_{x, y, t} = sum_{0}^{\infinite}{ sum_{0}^{\infinite}{ core } }
    
    assume a = N, b = M; x in [0, N], y in [0, M]; n in [0, N], m in [0, M]; with some slight change
    => 
    (\phi(x, y) = linear(dwconv(input(x, y))))
    A(n, m) = DCT2D(\phi(x, y))
    u(x, y, t) = IDCT2D(A(n, m) * (1 - [(n\pi/a)^2 + (m\pi/b)^2]c2t2) * e^(-αt))
    """    
    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )
        # Add wave speed parameter
        self.c = nn.Parameter(torch.ones(1) * 1)
        # Add damping parameter
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)
        
    def infer_init_wave2d(self, freq):
        weight_exp = self.get_decay_map((self.res, self.res), device=freq.device)
        self.k_exp = nn.Parameter(torch.pow(weight_exp[:, :, None], self.to_k(freq)), requires_grad=False)
        del self.to_k

    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):
        # cos((x + 0.5) / N * n * \pi) which is also the form of DCT and IDCT
        # DCT: F(n) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * f(x) )
        # IDCT: f(x) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * F(n) )
        # returns: (Res_n, Res_x)
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        # (1 - [(n\pi/a)^2 + (m\pi/b)^2]c2t2) * e^(-αt)
        # returns: (Res_h, Res_w)
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        # Quadratic term for wave equation
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight

    def forward(self, x: torch.Tensor, freq_embed=None):
        B, C, H, W = x.shape
        x = self.dwconv(x)
        x = self.linear(x.permute(0, 2, 3, 1).contiguous())  # B,H,W,2C
        x, z = x.chunk(chunks=2, dim=-1)  # B,H,W,C
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        z = z.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (weight_cosn is not None) and (weight_cosn.device == x.device):
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
        else:
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        def dct2d(x):
            """2D DCT-II on last two dims (H, W) - 改用torch_dct实现"""
            # torch_dct.dct2d默认是ortho归一化，对应type=2的DCT
            x = torch_dct.dct_2d(x, norm='ortho')
            return x

        def idct2d(x):
            """2D IDCT-II on last two dims (H, W) - 改用torch_dct实现"""
            # torch_dct.idct2d对应IDCT-II，与DCT2D配对
            x = torch_dct.idct_2d(x, norm='ortho')
            return x

        x_u0 = dct2d(x)
        x_v0 = dct2d(z)  # 修正笔误：原代码是dct2d(x)，应为z

        # freq_embed: (H, W, C) -> (B, H, W, C)
        if freq_embed is not None:
            t = self.to_k(freq_embed.unsqueeze(0).expand(B, -1, -1, -1).contiguous())
        else:
            t = torch.zeros((B, H, W, C), device=x.device, dtype=x.dtype)
        cos_term = torch.cos(self.c * t).permute(0, 3, 1, 2).contiguous()
        sin_term = torch.sin(self.c * t).permute(0, 3, 1, 2).contiguous() / self.c

        wave_term = cos_term * x_u0
        velocity_term = sin_term * (x_v0 + (self.alpha / 2) * x_u0)
        final_term = wave_term + velocity_term

        x_final = idct2d(final_term)
        # x_final: (B, C, H, W)
        x = self.out_norm(x_final.permute(0, 2, 3, 1).contiguous())
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x * F.silu(z)
        x = self.out_linear(x.permute(0, 2, 3, 1).contiguous())
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    


# 使用示例
if __name__ == "__main__":
    # 先安装依赖：pip install torch-dct
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    input_tensor = torch.randn(1, 96, 14, 14).to(device)
    wpo = WPO(res=14, dim=96, hidden_dim=96).to(device)

    print("WPO 模型结构:\n", wpo)

    output_tensor = wpo(input_tensor)

    # 打印维度验证
    print("\n输入张量形状  :", input_tensor.shape)   
    print("输出张量形状 :", output_tensor.shape) 
    print("\n代码运行成功！")