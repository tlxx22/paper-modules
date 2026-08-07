import torch
import torch.nn as nn

class RBFLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_min: float = -2,
        grid_max: float = 2,
        num_grids: int = 8,
        spline_weight_init_scale: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids


        base_grid = torch.linspace(grid_min, grid_max, num_grids)
        mu_init = base_grid.repeat(in_features)           

        sigma_init = torch.full((in_features * num_grids,), (base_grid[1] - base_grid[0]).abs() + 1e-6)


        self.mu = nn.Parameter(mu_init)
        self.log_sigma = nn.Parameter(sigma_init.log())


        self.spline_weight = nn.Parameter(
            torch.randn(in_features * num_grids, out_features) * spline_weight_init_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F = x.shape
        G = self.num_grids

        x_flat = x.reshape(B * S, F)
        mu = self.mu.view(1, F, G).to(dtype=x.dtype, device=x.device)
        sigma = self.log_sigma.exp().view(1, F, G).to(dtype=x.dtype, device=x.device) + 1e-6


        x_exp = x_flat.unsqueeze(-1)
        basis = torch.exp(-0.5 * ((x_exp - mu) / sigma) ** 2)
        basis = basis.reshape(B * S, F * G)

        out = basis @ self.spline_weight
        return out.view(B, S, self.out_features)


class RBFKANLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        grid_min: float = -2,
        grid_max: float = 2,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation=nn.SiLU(), 
        spline_weight_init_scale: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_base_update = use_base_update

        self.base_activation = type(base_activation)() if isinstance(base_activation, nn.Module) else nn.SiLU()                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!

        self.rbf_linear = RBFLinear(
            input_dim, output_dim,
            grid_min=grid_min, grid_max=grid_max,
            num_grids=num_grids,
            spline_weight_init_scale=spline_weight_init_scale
        )
        if use_base_update:
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.rbf_linear(x)
        if self.use_base_update:
            y = y + self.base_linear(self.base_activation(x))
        return y
    
class MultiPatchRBFKAN(nn.Module):
    def __init__(
        self,
        patch_size: int = 7,
        input_channels: int = 1,
        image_size=None,
        use_global_linear: bool = True
    ):
        super().__init__()
        self.ps = patch_size
        f = patch_size * patch_size
        self.patch_mixer = RBFKANLayer(
            input_dim=f, output_dim=f, num_grids=8,
            use_base_update=True, base_activation=nn.SiLU()
        )

        self.use_separable = True
        self.sep_kernel = 3
        self._sep_built_for_c = None
        self.dw_h = None
        self.dw_w = None
        self.reweight_head = None


        self.use_global_linear = use_global_linear
        self.global_rank_default = 64 
        self._lowrank_built_for_hw = None
        self.lin1 = None
        self.lin2 = None
        self.H = None
        self.W = None


    def initialize(self, H: int, W: int, device=None):
        self.H, self.W = H, W 

    def _build_separable(self, C: int, device):
        if self._sep_built_for_c == C:
            return
        k = self.sep_kernel
        self.dw_h = nn.Conv2d(C, C, kernel_size=(k, 1), padding=(k // 2, 0), groups=C).to(device)                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!
        self.dw_w = nn.Conv2d(C, C, kernel_size=(1, k), padding=(0, k // 2), groups=C).to(device)                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!
        self.reweight_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, max(8, C // 4), 1), nn.GELU(),
            nn.Conv2d(max(8, C // 4), 2, 1),
            nn.Softmax(dim=1)
        ).to(device)
        self._sep_built_for_c = C

 
    def _build_lowrank(self, H: int, W: int, device):
        if self._lowrank_built_for_hw == (H, W):
            return
        N = H * W
        r = min(self.global_rank_default, N)
        self.lin1 = nn.Linear(N, r, bias=False).to(device)
        self.lin2 = nn.Linear(r, N, bias=False).to(device)
        self._lowrank_built_for_hw = (H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ps = self.ps
        assert H % ps == 0 and W % ps == 0, f"H/W has to be devided by patch_size={ps}"                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!


        patches = x.unfold(2, ps, ps).unfold(3, ps, ps) 
        Hn, Wn = H // ps, W // ps
        tokens = patches.contiguous().view(B * C, Hn * Wn, ps * ps)

        mixed = self.patch_mixer(tokens)
        mixed = mixed + tokens


        mixed = mixed.view(B, C, Hn, Wn, ps, ps).permute(0, 1, 2, 4, 3, 5).contiguous()                                                                                                                                                                                            # 哔哩1哔哩/微信2公众号: A-I-缝-合-术, AI-Feng1-he2-shu3, 缝1-合2-术3-AI, AI1f-eng-hes-hu独家整理!
        y = mixed.view(B, C, H, W)


        if self.use_separable:
            self._build_separable(C, x.device)
            y_h = self.dw_h(y)
            y_w = self.dw_w(y)
            w = self.reweight_head(y)
            y = w[:, 0:1] * y_h + w[:, 1:2] * y_w

        if self.use_global_linear:
            self._build_lowrank(H, W, x.device)
            y_flat = y.view(B * C, -1)
            y_global = self.lin2(self.lin1(y_flat)).view(B, C, H, W)
            y = y + y_global

        return y


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 32, 49, 49).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = MultiPatchRBFKAN(patch_size=7, input_channels=32).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")