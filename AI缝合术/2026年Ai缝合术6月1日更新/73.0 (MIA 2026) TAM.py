import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Attention mechanism applied on a set of frames using Query, Key, and Value (QKV) attention.
    The mechanism computes attention for each frame using other frames as context, applies gating,
    and combines the features for further processing.

    Args:
        all_channels (int): Number of input channels for the frames (default: 1024).
        embedding_dim (int): The dimension of the embeddings used for attention (default: 1024).
        num_heads (int): Number of attention heads for multi-head attention (default: 8).
        dropout (float): Dropout rate for attention (default: 0.0).
    """
    
    def __init__(self, all_channels=1024, embedding_dim=1024, num_heads=8, dropout=0.0):                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        super().__init__()
        self.all_channels = all_channels
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # 注意：nn.MultiheadAttention内部已经包含了QKV投影
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=dropout
        )

        # 通道投影层：当embedding_dim != all_channels时使用
        if embedding_dim != all_channels:
            self.in_proj = nn.Linear(all_channels, embedding_dim, bias=False)
            self.out_proj = nn.Linear(embedding_dim, all_channels, bias=False)
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()

        # Gating mechanism (using convolution)
        self.gate_conv = nn.Conv2d(all_channels, 1, kernel_size=1, bias=False)
        self.gate_activation = nn.Sigmoid()

        # Convolutional layer for combining attention and original features
        self.combine_conv = nn.Conv2d(all_channels * 2, all_channels, kernel_size=3, padding=1, bias=False)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        # 使用LayerNorm代替BatchNorm，更适合序列数据
        self.norm = nn.LayerNorm(all_channels)
        self.activation = nn.ReLU(inplace=True)

        # Final classifier to process the output
        self.classifier = nn.Conv2d(all_channels, all_channels, kernel_size=1, bias=True)

    def forward(self, frames):
        """
        Forward pass through the attention mechanism for a set of frames.

        Args:
            frames (torch.Tensor): Input tensor of shape (B, C, H, W, T)
                where B=batch, C=channels, H=height, W=width, T=num_frames

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W, T)
        """
        B, C, H, W, T = frames.shape
        assert C == self.all_channels, f"输入通道数{C}与模型期望通道数{self.all_channels}不匹配"

        # 第一步：重塑张量为适合注意力计算的形状
        # (B, C, H, W, T) -> (B, T, H*W, C)
        x = frames.permute(0, 4, 2, 3, 1).reshape(B, T, H*W, C)
        
        # 第二步：计算所有帧对之间的注意力（向量化实现，无循环）
        # (B, T, H*W, C) -> (B*T, H*W, C)
        x_flat = x.reshape(B*T, H*W, C)
        
        # 投影到注意力维度
        x_proj = self.in_proj(x_flat)  # (B*T, H*W, E)
        
        # 计算自注意力：每个空间位置都能看到所有帧的对应位置
        # 注意：我们将T作为序列长度，H*W作为batch维度
        # (B*H*W, T, E)
        x_for_attn = x_proj.reshape(B, T, H*W, self.embedding_dim).permute(0, 2, 1, 3).reshape(B*H*W, T, self.embedding_dim)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        
        # 应用多头注意力
        attn_output, _ = self.self_attention(x_for_attn, x_for_attn, x_for_attn)  # (B*H*W, T, E)
        
        # 恢复形状
        attn_output = attn_output.reshape(B, H*W, T, self.embedding_dim).permute(0, 2, 1, 3)  # (B, T, H*W, E)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        attn_output = self.out_proj(attn_output)  # (B, T, H*W, C)
        
        # 第三步：应用门控机制
        # 恢复空间维度
        attn_output = attn_output.reshape(B, T, H, W, C).permute(0, 4, 2, 3, 1)  # (B, C, H, W, T)
        
        # 对每个时间步应用门控
        gated_outputs = []
        for t in range(T):
            attn_t = attn_output[:, :, :, :, t]  # (B, C, H, W)
            mask = self.gate_activation(self.gate_conv(attn_t))  # (B, 1, H, W)
            gated_t = attn_t * mask
            gated_outputs.append(gated_t)
        
        gated_output = torch.stack(gated_outputs, dim=-1)  # (B, C, H, W, T)
        
        # 第四步：结合原始特征
        combined = torch.cat([gated_output, frames], dim=1)  # (B, 2C, H, W, T)
        
        # 对每个时间步应用卷积和归一化
        final_outputs = []
        for t in range(T):
            combined_t = combined[:, :, :, :, t]  # (B, 2C, H, W)
            out_t = self.combine_conv(combined_t)  # (B, C, H, W)
            
            # 应用LayerNorm (在通道维度上)
            out_t = out_t.permute(0, 2, 3, 1)  # (B, H, W, C)
            out_t = self.norm(out_t)
            out_t = out_t.permute(0, 3, 1, 2)  # (B, C, H, W)
            
            out_t = self.activation(out_t)
            
            # 残差连接
            out_t = out_t + frames[:, :, :, :, t]
            
            # 最终分类器
            out_t = self.classifier(out_t)
            final_outputs.append(out_t)
        
        # 堆叠所有时间步的输出
        output = torch.stack(final_outputs, dim=-1)  # (B, C, H, W, T)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        
        return output


# 使用示例
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 输入形状: (batch_size, channels, height, width, time)
    input_tensor = torch.randn(1, 256, 16, 16, 10).to(device)

    # 初始化模型
    model = TemporalAttention(
        all_channels=256, 
        embedding_dim=256, 
        num_heads=8,
        dropout=0.1
    ).to(device)
    
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("\n维度验证:")
    print("输入形状:", input_tensor.shape)   
    print("输出形状:", output_tensor.shape)
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")