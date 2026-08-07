import torch
import torch.nn as nn

class SGILA(nn.Module):
    """
    Spatial Global Information Learning Attention (SGILA)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=1, 
            out_channels=1, 
            kernel_size=kernel_size, 
            padding=padding, 
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        # 1. 通道维度4种池化操作
        avg_pool = torch.mean(x, dim=1, keepdim=True)    # AP: 平均池化                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # MP: 最大池化                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
        min_pool, _ = torch.min(x, dim=1, keepdim=True)  # MN: 最小池化                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
        sum_pool = torch.sum(x, dim=1, keepdim=True)     # SP: 求和池化                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!

        # 2. 逐元素相加（对应结构图⊕操作）
        pool_aggregate = avg_pool + max_pool + min_pool + sum_pool                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!

        # 3. 7x7卷积 + Sigmoid 生成归一化的空间注意力图
        spatial_feat = self.conv(pool_aggregate)
        attention_map = self.sigmoid(spatial_feat)

        # 4. 注意力图与原始输入逐元素相乘（对应结构图⊗操作），输出加权后的特征                                                                                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu, 缝合术AI, AIfengheshu独家整理!
        output = x * attention_map

        return output

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(1, 64, 128, 128).to(device)

    model = SGILA().to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape) 
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")