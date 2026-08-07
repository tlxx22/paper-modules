import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfCrossAttention(nn.Module):
    """
    自-交叉注意力（改进到2D，适用于图像处理/计算机视觉任务）
    输入: x (B, C, H, W)
    输出: y (B, C, H, W)
    """
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0.):                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 1. 自注意力部分
        self.self_qkv = nn.Linear(dim, dim * 3)
        self.self_attn_drop = nn.Dropout(attn_drop)
        self.self_proj = nn.Linear(dim, dim)
        self.self_proj_drop = nn.Dropout(proj_drop)

        # 2. Split 分支
        self.split_q = nn.Linear(dim, dim)  # 共享查询 Q'
        self.split_kv1 = nn.Linear(dim, dim * 2)  # 绿分支 Kp, Vp                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.split_kv2 = nn.Linear(dim, dim * 2)  # 红分支 Kp, Vp

        # 3. 交叉注意力投影
        self.cross_proj1 = nn.Linear(dim, dim)
        self.cross_proj2 = nn.Linear(dim, dim)
        self.cross_proj_drop = nn.Dropout(proj_drop)

        # 最终融合
        self.out_proj = nn.Linear(dim, dim)
        self.out_proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        # 展平为序列
        x_flat = x.flatten(2).transpose(1, 2)  # (B, N, C)

        # ---------------------- 1. 自注意力（带残差） ----------------------
        qkv = self.self_qkv(x_flat).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.self_attn_drop(attn)

        self_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        self_out = self.self_proj(self_out)
        self_out = self.self_proj_drop(self_out)

        # 自注意力残差连接
        x_self = x_flat + self_out  # (B, N, C)

        # ---------------------- 2. Split 分支（共享Q' + 两个独立分支） ----------------------
        # 共享查询 Q'
        q_prime = self.split_q(x_self).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, heads, N, head_dim)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        # 绿分支 Kp, Vp
        kv1 = self.split_kv1(x_self).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        k1, v1 = kv1[0], kv1[1]  # (B, heads, N, head_dim)

        # 红分支 Kp, Vp
        kv2 = self.split_kv2(x_self).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        k2, v2 = kv2[0], kv2[1]  # (B, heads, N, head_dim)

        # ---------------------- 3. 交叉注意力分支（并行两个） ----------------------
        # 交叉注意力1（绿分支）
        attn1 = (q_prime @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        cross_out1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        cross_out1 = self.cross_proj1(cross_out1)
        cross_out1 = self.cross_proj_drop(cross_out1)
        cross_out1 = cross_out1 + x_self  # 残差连接

        # 交叉注意力2（红分支）
        attn2 = (q_prime @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        cross_out2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        cross_out2 = self.cross_proj2(cross_out2)
        cross_out2 = self.cross_proj_drop(cross_out2)
        cross_out2 = cross_out2 + x_self  # 残差连接

        # ---------------------- 4. 融合两个交叉分支输出 + 最终残差 ----------------------                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        cross_fusion = cross_out1 + cross_out2
        out = self.out_proj(cross_fusion)
        out = self.out_proj_drop(out)

        # 整体残差（对应图最上方的加法）
        out = out + x_self

        # 恢复为 2D 特征图
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 16, 32, 32).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = SelfCrossAttention(dim=16).to(device)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print(model)
    
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")