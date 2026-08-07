import torch
import torch.nn as nn

class ComponentWiseAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(ComponentWiseAttention, self).__init__()
        # 深度卷积 (DWConv)：保持通道独立性，7x7 卷积核
        self.dwconv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,  # 保持特征图尺寸不变
            groups=in_channels,      # 深度卷积：每个通道单独卷积                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            bias=False
        )
        # 1x1 卷积：通道对齐，零偏置
        self.conv1x1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,  # 输出通道数与输入一致 (C9 = C)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            kernel_size=1,
            bias=False  # 零偏置，减少参数
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 步骤1：深度卷积 (DWConv)
        x_dw = self.dwconv(x)
        # 步骤2：1x1 卷积 (通道对齐)
        x_conv = self.conv1x1(x_dw)
        # 步骤3：Sigmoid 激活生成注意力图
        attention_map = self.sigmoid(x_conv)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        return attention_map

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 32, 256, 256).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = ComponentWiseAttention(in_channels=32, kernel_size=7).to(device)                                                                                                                                                                                           # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print(model)
    
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")