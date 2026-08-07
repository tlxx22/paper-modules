import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)                                                                                                                             # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):                                                                                                                             # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )
    def forward(self, x):
        return x * self.scale
    
class FFN_DIFF(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.667, bias=False):
        super(FFN_DIFF, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.sigma = ElementScale(
            hidden_features//4, init_value=1e-5, requires_grad=True)
        self.decompose = nn.Conv2d(
            in_channels=hidden_features//4,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.decompose_act = nn.GELU()
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv_5 = nn.Conv2d(hidden_features//4, hidden_features//4, kernel_size=5, 
                                stride=1, padding=2, groups=hidden_features//4, bias=bias)
        self.dwconv_dilated2_1 = nn.Conv2d(hidden_features//4, hidden_features//4, kernel_size=3,                                                                                                                              # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
                                         stride=1, padding=2, groups=hidden_features//4, 
                                         bias=bias, dilation=2)
        self.p_unshuffle = nn.PixelUnshuffle(2)
        self.p_shuffle = nn.PixelShuffle(2)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x
    
    def forward(self, x):
        x = self.project_in(x)
        x = self.p_shuffle(x)
        x = channel_shuffle(x, groups=1) 
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv_5(x1)
        x2 = self.dwconv_dilated2_1(x2)
        x = F.mish(x2) * x1
        x = self.feat_decompose(x)
        x = self.p_unshuffle(x)
        x = self.project_out(x)
        return x

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    model = FFN_DIFF(dim=3).to(device)

    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape) 
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")