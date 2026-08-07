import torch
import torch.nn as nn

class GCA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCA, self).__init__()
        self.normal = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        b, c, w, h = x.shape[0], x.shape[1], x.shape[2], x.shape[3]                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        avepool = nn.AdaptiveAvgPool2d((1, 1))
        maxpool = nn.AdaptiveMaxPool2d((1, 1))
        x_g = avepool(x) + maxpool(x)
        z_g_out = torch.Tensor().cuda()
        for i in range(b):
            x_g_i = x_g[i].view([c])
            _, index = torch.topk(x_g_i, 1)
            mean = index[0] * 1.0  # 均值
            covariance = (c / 2) ** 2  # 协方差
            gaussian_distribution = torch.distributions.Normal(mean, covariance)                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
            x_g_index = torch.linspace(0, c, c).cuda()
            z_g = gaussian_distribution.log_prob(x_g_index.flatten())                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
            z_g = z_g.exp().reshape(x_g_index.shape)
            z_g = z_g.view([1, c, 1, 1])
            z_g_out = torch.cat((z_g_out, z_g), dim=0)
        z_g_out = self.normal(z_g_out)
        return x * z_g_out

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 输入特征图 
    input_tensor = torch.randn(2, 32, 256, 256).to(device)    # (batch_size, channels, height, width)                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!

    model = GCA(32, 32).to(device)

    print(model)

    # 前向传播
    output_tensor = model(input_tensor)

    # 打印输入输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")