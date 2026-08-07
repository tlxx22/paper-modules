import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


class CGILA(nn.Module):
    """
    Channel Global Information Learning Attention (CGILA)
    通道全局信息学习注意力模块，兼容图像分类、目标检测、语义分割等所有CV任务                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
    """
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 通道映射全连接层（用1x1卷积替代Linear，无需reshape）
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, bias=True)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 四分支并行提取不同维度的全局通道信息
        gap = self.avg_pool(x)          # 全局平均池化：捕捉通道整体分布
        gmp = self.max_pool(x)          # 全局最大池化：捕捉通道最强响应
        gmn = -self.max_pool(-x)        # 全局最小池化：捕捉通道最弱响应（复用max_pool）                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
        gsp = torch.sum(x, dim=(2, 3), keepdim=True)  # 全局求和池化：捕捉通道总能量                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
        
        # 2. 多源全局特征逐元素融合
        fusion_feat = gap + gmp + gmn + gsp
        
        # 3. 学习通道间依赖关系，生成注意力权重
        att_map = self.fc(fusion_feat)
        att_map = self.sigmoid(att_map)  # 权重归一化到[0,1]
        
        # 4. 原始特征与注意力图逐元素相乘，得到加权输出
        out = x * att_map
        
        return out


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 64, 32, 32).to(device)

    model = CGILA(64).to(device)

    print(model)
    
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape) 
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")