import torch
import torch.nn as nn
import torch.nn.functional as F


class DFE(nn.Module):
    """ Dual Feature Extraction 
    Args:
        in_features (int): Number of input channels.
        out_features (int): Number of output channels.
    """
    def __init__(self, in_features, out_features):
        super().__init__()

        self.out_features = out_features

        self.conv = nn.Sequential(nn.Conv2d(in_features, in_features // 5, 1, 1, 0),                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(in_features // 5, in_features // 5, 3, 1, 1),                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                        nn.Conv2d(in_features // 5, out_features, 1, 1, 0))
        
        self.linear = nn.Conv2d(in_features, out_features,1,1,0)

    def forward(self, x, x_size):
        
        B, L, C = x.shape
        H, W = x_size
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        x = self.conv(x) * self.linear(x)
        x = x.view(B, -1, H*W).permute(0,2,1).contiguous()

        return x
    
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (tuple): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (tuple): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] * (window_size[0] * window_size[1]) / (H * W))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of heads for spatial self-correlation.
        residual (bool):  If True, use residual strage to connect conv.
    """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class SCC(nn.Module):
    """ Spatial-Channel Correlation.
    Args:
        dim (int): Number of input channels.
        base_win_size (tuple[int]): The height and width of the base window.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of heads for spatial self-correlation.
        value_drop (float, optional): Dropout ratio of value. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, base_win_size, window_size, num_heads, value_drop=0., proj_drop=0.):

        super().__init__()
        # parameters
        self.dim = dim
        self.window_size = window_size 
        self.num_heads = num_heads

        # feature projection
        self.qv = DFE(dim, dim)
        self.proj = nn.Linear(dim, dim)

        # dropout
        self.value_drop = nn.Dropout(value_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # base window size
        min_h = min(self.window_size[0], base_win_size[0])
        min_w = min(self.window_size[1], base_win_size[1])
        self.base_win_size = (min_h, min_w)

        # normalization factor and spatial linear layer for S-SC
        head_dim = dim // (2*num_heads)
        self.scale = head_dim
        self.spatial_linear = nn.Linear(self.window_size[0]*self.window_size[1] // (self.base_win_size[0]*self.base_win_size[1]), 1)

        # define a parameter table of relative position bias
        self.H_sp, self.W_sp = self.window_size
        self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
    
    def spatial_linear_projection(self, x):
        B, num_h, L, C = x.shape
        H, W = self.window_size
        map_H, map_W = self.base_win_size

        x = x.view(B, num_h, map_H, H//map_H, map_W, W//map_W, C).permute(0,1,2,4,6,3,5).contiguous().view(B, num_h, map_H*map_W, C, -1)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        x = self.spatial_linear(x).view(B, num_h, map_H*map_W, C)
        return x
    
    def spatial_self_correlation(self, q, v):
        
        B, num_head, L, C = q.shape

        # spatial projection
        v = self.spatial_linear_projection(v)

        # compute correlation map
        corr_map = (q @ v.transpose(-2,-1)) / self.scale

        # add relative position bias
        # generate mother-set
        position_bias_h = torch.arange(1 - self.H_sp, self.H_sp, device=v.device)
        position_bias_w = torch.arange(1 - self.W_sp, self.W_sp, device=v.device)
        biases = torch.stack(torch.meshgrid(position_bias_h, position_bias_w, indexing='ij'))
        rpe_biases = biases.flatten(1).transpose(0, 1).contiguous().float()
        pos = self.pos(rpe_biases)

        # select position bias
        coords_h = torch.arange(self.H_sp, device=v.device)
        coords_w = torch.arange(self.W_sp, device=v.device)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.H_sp - 1
        relative_coords[:, :, 1] += self.W_sp - 1
        relative_coords[:, :, 0] *= 2 * self.W_sp - 1
        relative_position_index = relative_coords.sum(-1)
        relative_position_bias = pos[relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.base_win_size[0], self.window_size[0]//self.base_win_size[0], self.base_win_size[1], self.window_size[1]//self.base_win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(0,1,3,5,2,4).contiguous().view(
            self.window_size[0] * self.window_size[1], self.base_win_size[0]*self.base_win_size[1], self.num_heads, -1).mean(-1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() 
        corr_map = corr_map + relative_position_bias.unsqueeze(0)

        # transformation
        v_drop = self.value_drop(v)
        x = (corr_map @ v_drop).permute(0,2,1,3).contiguous().view(B, L, -1) 

        return x
    
    def channel_self_correlation(self, q, v):
        
        B, num_head, L, C = q.shape

        # apply single head strategy
        q = q.permute(0,2,1,3).contiguous().view(B, L, num_head*C)
        v = v.permute(0,2,1,3).contiguous().view(B, L, num_head*C)

        # compute correlation map
        corr_map = (q.transpose(-2,-1) @ v) / L
        
        # transformation
        v_drop = self.value_drop(v)
        x = (corr_map @ v_drop.transpose(-2,-1)).permute(0,2,1).contiguous().view(B, L, -1)

        return x

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, C, H, W)
        """
        x = x.permute(0,2,3,1).contiguous()
        xB,xH,xW,xC = x.shape
        qv = self.qv(x.view(xB,-1,xC), (xH,xW)).view(xB, xH, xW, xC)
        # window partition
        qv = window_partition(qv, self.window_size)
        qv = qv.view(-1, self.window_size[0]*self.window_size[1], xC)

        # qv splitting
        B, L, C = qv.shape
        qv = qv.view(B, L, 2, self.num_heads, C // (2*self.num_heads)).permute(2,0,3,1,4).contiguous()
        q, v = qv[0], qv[1]  # B, num_heads, L, C//num_heads

        # spatial self-correlation (S-SC)
        x_spatial = self.spatial_self_correlation(q, v)
        x_spatial = x_spatial.view(-1, self.window_size[0], self.window_size[1], C//2)
        x_spatial = window_reverse(x_spatial, (self.window_size[0],self.window_size[1]), xH, xW)  # xB xH xW xC

        # channel self-correlation (C-SC)
        x_channel = self.channel_self_correlation(q, v)
        x_channel = x_channel.view(-1, self.window_size[0], self.window_size[1], C//2)
        x_channel = window_reverse(x_channel, (self.window_size[0], self.window_size[1]), xH, xW) # xB xH xW xC

        # spatial-channel information fusion
        x = torch.cat([x_spatial, x_channel], -1)
        x = self.proj_drop(self.proj(x))

        x = x.permute(0,3,2,1).contiguous()
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 32, 128, 128).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = SCC(dim=32, base_win_size=(16, 16), window_size=(16, 16), num_heads=4).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")