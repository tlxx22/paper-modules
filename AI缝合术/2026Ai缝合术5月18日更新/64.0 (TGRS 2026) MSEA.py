import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """CBAM通道注意力"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)


class SpatialAttention(nn.Module):
    """CBAM空间注意力"""
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x)


class DenseDilatedContext(nn.Module):
    """密集连接空洞卷积多尺度上下文提取模块
   3个空洞卷积，膨胀率[3,5,7]
    """
    def __init__(self, in_channels, dilations=[3, 5, 7]):
        super().__init__()
        self.dilations = dilations
        self.num_dilations = len(dilations)
        
        # 密集连接的空洞卷积层
        self.convs = nn.ModuleList()
        for i in range(self.num_dilations):
            in_ch = in_channels * (i + 1)  # 每个层接收前面所有层的输出
            out_ch = in_channels
            self.convs.append(
                nn.Conv2d(in_ch, out_ch, 3, 
                          padding=dilations[i], 
                          dilation=dilations[i], 
                          bias=False)
            )
        
        # 1×1卷积降维，将拼接后的4C通道还原为C通道
        self.bottleneck = nn.Conv2d(
            in_channels * (self.num_dilations + 1), 
            in_channels, 1, bias=False
        )
        
    def forward(self, x):
        features = [x]
        for i in range(self.num_dilations):
            concat_feat = torch.cat(features, dim=1)
            out = self.convs[i](concat_feat)
            features.append(out)
        
        # 拼接所有尺度特征并降维
        concat_all = torch.cat(features, dim=1)
        out = self.bottleneck(concat_all)
        return out


class EdgeFocusedIntegrationAttention(nn.Module):
    """边缘聚焦集成注意力(EFIA)
    将边缘信息显式嵌入到空间注意力
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention()
        
        # 边缘预测分支
        self.edge_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            nn.Conv2d(in_channels, 1, 3, padding=1, bias=False)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        )
        
    def forward(self, x):
        # 步骤1: 通道注意力加权
        ca_map = self.ca(x)
        f_ca = x * ca_map
        
        # 步骤2: 边缘分支预测
        a_pred = self.edge_branch(f_ca)  # 输出边缘预测图
        
        # 步骤3: 空间注意力计算
        sa_map = self.sa(f_ca)
        
        # 步骤4: 边缘信息嵌入到空间注意力
        a_sig = torch.sigmoid(a_pred)
        a_edge = a_sig + sa_map  # 逐元素相加融合
        
        # 步骤5: 边缘增强特征输出
        f_efia = f_ca * a_edge
        
        return f_efia, a_pred


class MSEA(nn.Module):
    """多尺度边缘感知注意力(MSEA)
    于增强红外小目标的边缘特征
    """
    def __init__(self, in_channels, reduction=16, dilations=[3, 5, 7]):
        super().__init__()
        # 多尺度上下文提取
        self.dense_dilated = DenseDilatedContext(in_channels, dilations)
        
        # 边缘聚焦集成注意力
        self.efia = EdgeFocusedIntegrationAttention(in_channels, reduction)
        
        # 特征细化层
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            nn.BatchNorm2d(in_channels)
        )
        
        # 残差连接激活
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        """
        参数:
            x: 输入特征图，形状为[B, C, H, W]
        返回:
            out: 边缘增强后的特征图，形状与输入相同
            a_pred: 边缘预测图，形状为[B, 1, H, W]，用于边缘监督损失计算                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        """
        # 1. 多尺度上下文提取
        f_msce = self.dense_dilated(x)
        
        # 2. 边缘聚焦注意力增强
        f_efia, a_pred = self.efia(f_msce)
        
        # 3. 特征细化
        f_refine = self.refine(f_efia)
        
        # 4. 残差连接
        out = f_refine + x
        out = self.relu(out)
        
        return out, a_pred

def generate_edge_gt(gt, device='cuda'):
    """
    从目标分割GT生成边缘GT
    参数:
        gt: 目标分割GT，形状为[B, 1, H, W]，值为0或1
        device: 计算设备
    返回:
        edge_gt: 边缘GT，形状为[B, 1, H, W]，值为0或1
    """
    # 定义Sobel算子
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
        dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)
    
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
        dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)
    
    # 计算梯度
    grad_x = F.conv2d(gt, sobel_x, padding=1)
    grad_y = F.conv2d(gt, sobel_y, padding=1)
    
    # 计算梯度幅值并二值化
    edge_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    edge_gt = (edge_magnitude > 0).float()
    
    return edge_gt

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 32, 256, 256).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = MSEA(in_channels=32).to(device)
    print(model)
    # edge_pred 用于边缘预测图都进行 BCE 损失监督
    enhanced_feat, edge_pred = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print(f"增强后特征形状: {enhanced_feat.shape}")
    print(f"边缘预测图形状: {edge_pred.shape}")                                                                                                                                                                                         # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")