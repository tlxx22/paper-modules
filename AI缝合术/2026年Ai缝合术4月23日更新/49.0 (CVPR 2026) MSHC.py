import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelShuffle(nn.Module):
    """
    通道混洗（Channel Shuffle）实现
    用于打破分组卷积带来的通道信息孤岛
    """
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        # 步骤1: 将通道维度重塑为 (groups, channels//groups)
        x = x.view(batch_size, self.groups, channels // self.groups, height, width)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        # 步骤2: 转置两个分组维度
        x = x.transpose(1, 2).contiguous()
        # 步骤3: 展平回原始通道维度
        x = x.view(batch_size, channels, height, width)
        return x

class MSHC(nn.Module):
    """
    多尺度空间异构卷积（Multi-scale Spatial Heterogeneous Convolution）
    输入: x_i (B, C, H, W)
    输出: x_i' (B, C, H, W)  # 通道数与空间尺寸均保持不变
    """
    def __init__(self, in_channels, groups=4, dilated_rate=2):
        """
        参数:
            in_channels: 输入通道数
            groups: 分组数（用于GPC和Channel Shuffle）
            dilated_rate: DDC分支的空洞率
        """
        super().__init__()
        self.groups = groups
        
        # --------------------------
        # 1. 多尺度异构分支 (Parallel Branches)
        # --------------------------
        
        # 分支1: 1x1 分组逐点卷积 (GPC)
        self.branch_gpc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 分支2: 3x3 深度卷积 (DWC)
        self.branch_dwc3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 分支3: 5x5 深度卷积 (DWC)
        self.branch_dwc5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False),                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 分支4: 3x3 空洞深度卷积 (DDC)
        # padding计算: dilation * (kernel_size-1) // 2，以保持空间尺寸不变
        self.branch_ddc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                      padding=dilated_rate, dilation=dilated_rate, 
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # --------------------------
        # 2. 特征融合与后处理
        # --------------------------
        self.channel_shuffle = ChannelShuffle(groups=groups)
        
        # 后续精炼模块 (CS -> DWC3 -> GPC -> SDWC)
        self.post_process = nn.Sequential(
            # 3x3 DWC
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # 1x1 GPC (分组逐点卷积)
            nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # SDWC (空间深度卷积，这里使用3x3深度卷积实现)
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        # 1. 多尺度分支并行提取
        out_gpc = self.branch_gpc(x)
        out_dwc3 = self.branch_dwc3(x)
        out_dwc5 = self.branch_dwc5(x)
        out_ddc = self.branch_ddc(x)
        
        # 2. 逐元素相加融合 (Element-wise Addition)
        out = out_gpc + out_dwc3 + out_dwc5 + out_ddc
        
        # 3. 通道混洗
        out = self.channel_shuffle(out)
        
        # 4. 最终特征精炼
        out = self.post_process(out)
        
        return out
    

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 32, 256, 256).to(device)

    image_size = 256

    model = MSHC(in_channels=32, groups=4, dilated_rate=2).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    print(model)
    
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")