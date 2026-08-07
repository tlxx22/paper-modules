import torch
import torch.nn as nn
import torch.nn.functional as F

class CAFR(nn.Module):
    def __init__(self, in_channels):
        super(CAFR, self).__init__()
        # 3x3卷积用于特征对齐（调整通道数与高层特征一致）
        self.align_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # 全连接层生成全局光谱向量
        self.fc = nn.Linear(in_channels, in_channels)

    def forward(self, high_level_feat, low_level_feat):
        # Args:
        #     high_level_feat: 高层特征 (B, C, H, W)
        #     low_level_feat: 低层特征 (B, C, 2H, 2W)
        # Returns:
        #     refined_feat: 细化后的特征 (B, C, H, W)
        
        # 1. 特征对齐：低层特征通过3x3卷积调整通道，并上采样至高层特征尺寸
        low_level_align = self.align_conv(low_level_feat)  # (B, C, 2H, 2W)
        low_level_align = F.interpolate(
            low_level_align, 
            size=high_level_feat.shape[2:],  # 上采样到高层特征的H×W
            mode="bilinear", 
            align_corners=False
        )  # (B, C, H, W)

        # 2. 计算全局光谱向量（GAP + 全连接）
        # 高层特征全局平均池化 + FC
        gap_high = F.adaptive_avg_pool2d(high_level_feat, (1, 1)).flatten(1)  # (B, C)
        Vh = self.fc(gap_high).unsqueeze(-1)  # (B, C, 1) → 高层向量 (C×1)
        
        # 低层特征全局平均池化 + FC
        gap_low = F.adaptive_avg_pool2d(low_level_align, (1, 1)).flatten(1)  # (B, C)
        Vl = self.fc(gap_low).unsqueeze(-1)  # (B, C, 1) → 低层向量 (C×1)

        # 3. 计算交叉光谱注意力权重
        A = torch.matmul(Vh, Vl.transpose(1, 2))  # (B,C,1) @ (B,1,C) → (B,C,C) 生成C×C注意力矩阵
        A = F.softmax(A, dim=-1)  # 沿最后一维归一化

        # 4. 生成融合权重
        W_h = torch.matmul(A.transpose(-1, -2), Vh)  # (B,C,C) @ (B,C,1) → (B,C,1)
        W_l = torch.matmul(A, Vl)  # (B,C,C) @ (B,C,1) → (B,C,1)
        
        # 扩展为4D张量以支持广播乘法
        W_h = W_h.unsqueeze(-1)  # (B,C,1,1)
        W_l = W_l.unsqueeze(-1)  # (B,C,1,1)
        
        # 5. 特征加权融合
        refined_feat = W_h * high_level_feat + W_l * low_level_align
        
        return refined_feat
    

# 使用示例
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 模拟高层和低层特征（batch_size=2，通道数=256）
    high_level = torch.randn(2, 256, 32, 32).to(device)  # 高层特征：32×32
    low_level = torch.randn(2, 256, 64, 64).to(device)   # 低层特征：64×64（尺寸为高层的2倍）
    
    cafr = CAFR(256).to(device)
    print(cafr)
    output_tensor = cafr(high_level, low_level)
    
    # 打印输入输出形状
    print(f"Input shapes:\n\tHigh-Level Feature Shape: {high_level.shape}\n\tLow-Level Feature Shape: {low_level.shape}")
    print(f"Output shape: {output_tensor.shape}")
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")