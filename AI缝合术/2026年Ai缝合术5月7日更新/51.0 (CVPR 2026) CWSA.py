import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelWiseSelfAttention(nn.Module):
    """
    通道自注意力模块 (Channel-wise Self-Attention)
    输入输出形状: [batch_size, channels, height, width]                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    计算复杂度: O(C²HW)，适合中低通道数的特征图
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 输入特征通道数C
        
        # Q/K/V线性投影 (1x1卷积，保持空间维度不变)
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        
        # 输出投影层
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape  # 输入形状: [B, C, H, W]
        N = H * W  # 空间像素总数
        
        # -------------------------- 1. Q/K/V投影 --------------------------
        q = self.q_proj(x)  # [B, C, H, W]
        k = self.k_proj(x)  # [B, C, H, W]
        v = self.v_proj(x)  # [B, C, H, W]
        
        # -------------------------- 2. 维度重塑 (对应流程图reshape) --------------------------                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        # 将空间维度展平，变为 [B, C, N] (N=H*W)，对应流程图的 H'W'×C
        q = q.reshape(B, C, N)  # [B, C, N]
        k = k.reshape(B, C, N)  # [B, C, N]
        v = v.reshape(B, C, N)  # [B, C, N]
        
        # -------------------------- 3. 计算通道注意力 --------------------------
        # Q转置后与K相乘，得到C×C的通道间相关性矩阵 (严格对应流程图)
        # (C×N) × (N×C) = C×C
        attn = torch.matmul(q, k.transpose(1, 2)) / (C ** 0.5)  # [B, C, C]
        attn = F.softmax(attn, dim=-1)  # 沿通道维度归一化
        
        # 注意力权重与V相乘，得到加权后的通道特征
        # (C×C) × (C×N) = C×N
        out = torch.matmul(attn, v)  # [B, C, N]
        
        # -------------------------- 4. 维度恢复与残差连接 --------------------------                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        # 恢复原始空间形状
        out = out.reshape(B, C, H, W)  # [B, C, H, W]
        
        # 输出投影
        out = self.out_proj(out)
        
        # 残差连接
        out = out + x  # [B, C, H, W]                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        
        return out


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 32, 256, 256).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = ChannelWiseSelfAttention(dim=32).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    print(model)
    
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")