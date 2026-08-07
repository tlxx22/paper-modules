import torch
import torch.nn as nn

class CFEM(nn.Module):
    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(1, in_channels // 2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.avg_conv_cat = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.max_conv_cat = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        self.avg_conv_sum = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.max_conv_sum = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.att_conv = nn.Conv2d(in_channels + 2 * hidden_channels, in_channels, kernel_size=1)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_feat = self.avg_pool(x)
        max_feat = self.max_pool(x)

        avg_cat = self.avg_conv_cat(avg_feat)
        max_cat = self.max_conv_cat(max_feat)
        fuse_cat = torch.cat([avg_cat, max_cat], dim=1)

        avg_sum = self.avg_conv_sum(avg_feat)
        max_sum = self.max_conv_sum(max_feat)
        fuse_sum = avg_sum + max_sum

        fuse = torch.cat([fuse_cat, fuse_sum], dim=1)
        att = self.sigmoid(self.att_conv(fuse))

        out = x * att
        return out

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 32, 128, 128).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = CFEM(in_channels=32).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")