import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class DWConvAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            input_height: int,
            input_width: int = None,
            num_heads: int = 8,
            qkv_bias: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            init_conv: bool = True,
            **kwargs
    ) -> None:
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width if input_width is not None else input_height  # 默认正方形特征图                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.num_heads = num_heads

        self.head_dim = dim // num_heads
        self.conv_dim = dim
        self.scale = self.head_dim ** -0.5

        # 深度可分离卷积 (groups=dim)
        self.qk_conv = nn.Conv2d(
            in_channels=self.conv_dim, 
            out_channels=self.conv_dim,
            groups=self.conv_dim, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=qkv_bias
        )

        # Value投影
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        # Dropout层
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        # 可选的高斯初始化
        if init_conv:
            self.init_conv()

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape

        # 计算预期的patch数量
        expected_patches = self.input_height * self.input_width
        
        # 形状检查
        if N != expected_patches:
            raise ValueError(
                f"序列长度N={N} 必须等于 input_height × input_width = {expected_patches}\n"                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
                f"当前配置: input_height={self.input_height}, input_width={self.input_width}"                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            )

        # Value投影
        v = self.v_proj(x)

        # 将序列形式转换为2D特征图形式
        v_conv = v.transpose(1, 2).reshape(B, self.conv_dim, self.input_height, self.input_width).contiguous()                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        # 应用深度卷积注意力
        x_conv = self.qk_conv(v_conv)

        # 将2D特征图转换回序列形式
        x = x_conv.reshape(B, self.conv_dim, N).transpose(1, 2)

        # 输出投影和Dropout
        x = self.attn_drop(x)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def init_conv(self, size=3, sigma=1.0):
        """将卷积核初始化为高斯核"""
        coords = torch.arange(size) - size // 2
        x, y = torch.meshgrid(coords, coords, indexing='ij')

        # 计算2D高斯函数
        kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        kernel = kernel / kernel.sum() 
        kernel = kernel.expand_as(self.qk_conv.weight).clone()
        self.qk_conv.weight = torch.nn.Parameter(kernel)


# 使用示例
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 输入形状: (batch_size, num_patches, channels)
    input_tensor = torch.randn(2, 1024, 32).to(device)

    # 初始化模型
    model = DWConvAttention(dim=32, input_height=32, input_width=32,num_heads=8, qkv_bias=False, proj_bias=True, attn_drop=0.0, proj_drop=0.0, init_conv=True).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("\n维度验证:")
    print("输入形状:", input_tensor.shape)   
    print("输出形状:", output_tensor.shape)
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")