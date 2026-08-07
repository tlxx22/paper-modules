import torch
import torch.nn as nn
import torch.nn.functional as F

class VerticalHorizontalCrossAttention(nn.Module):
    """
    垂直-水平交叉注意力模块 (Vertical-Horizontal Cross-Attention)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    输入输出形状: [batch_size, channels, height, width]
    计算复杂度: O(HW(H+W))，远低于标准全局注意力的O((HW)^2)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 输入特征通道数C
        
        # 共享的Q/K/V线性投影 (1x1卷积)
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        
        # 输出投影层
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    def forward(self, x):
        B, C, H, W = x.shape  # 输入形状: [B, C, H, W]
        
        # -------------------------- 1. 共享Q/K/V投影 --------------------------
        q = self.q_proj(x)  # [B, C, H, W]
        k = self.k_proj(x)  # [B, C, H, W]
        v = self.v_proj(x)  # [B, C, H, W]
        
        # -------------------------- 2. 垂直注意力分支 (Vertical-Attention) --------------------------
        # 转置H和W，将每一列作为独立注意力单元，W变为批量维度
        q_v = q.permute(0, 1, 3, 2)  # [B, C, W, H]
        k_v = k.permute(0, 1, 3, 2)  # [B, C, W, H]
        v_v = v.permute(0, 1, 3, 2)  # [B, C, W, H]
        
        # 重塑为注意力计算格式: [批量*列数, 行数, 通道数]
        q_v = q_v.reshape(B * W, C, H).transpose(1, 2)  # [B*W, H, C]
        k_v = k_v.reshape(B * W, C, H)                 # [B*W, C, H]                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        v_v = v_v.reshape(B * W, C, H).transpose(1, 2)  # [B*W, H, C]
        
        # 计算列内注意力分数 + 缩放防止梯度消失
        attn_v = torch.matmul(q_v, k_v) / (C ** 0.5)  # [B*W, H, H]
        attn_v = F.softmax(attn_v, dim=-1)
        
        # 加权求和得到垂直注意力输出
        out_v = torch.matmul(attn_v, v_v)  # [B*W, H, C]
        
        # 恢复原始形状
        out_v = out_v.transpose(1, 2).reshape(B, C, W, H)  # [B, C, W, H]                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        out_v = out_v.permute(0, 1, 3, 2)  # 转回原始空间顺序 [B, C, H, W]
        
        # -------------------------- 3. 水平注意力分支 (Horizontal-Attention) --------------------------                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        # 无需转置，将每一行作为独立注意力单元，H变为批量维度
        q_h = q.reshape(B * H, C, W).transpose(1, 2)  # [B*H, W, C]
        k_h = k.reshape(B * H, C, W)                 # [B*H, C, W]
        v_h = v.reshape(B * H, C, W).transpose(1, 2)  # [B*H, W, C]
        
        # 计算行内注意力分数 + 缩放
        attn_h = torch.matmul(q_h, k_h) / (C ** 0.5)  # [B*H, W, W]
        attn_h = F.softmax(attn_h, dim=-1)
        
        # 加权求和得到水平注意力输出
        out_h = torch.matmul(attn_h, v_h)  # [B*H, W, C]
        
        # 恢复原始形状
        out_h = out_h.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        
        # -------------------------- 4. 特征融合与残差连接 --------------------------                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        # 先融合垂直和水平注意力特征
        out = out_v + out_h  # [B, C, H, W]
        
        # 输出投影
        out = self.out_proj(out)
        
        # 最后添加残差连接
        out = out + x  # [B, C, H, W]                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        
        return out


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 32, 256, 256).to(device)

    model = VerticalHorizontalCrossAttention(dim=32).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    print(model)
    
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")