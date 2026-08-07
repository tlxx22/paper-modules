import torch
import torch.nn as nn

class GSA(nn.Module):
    def __init__(self, channels):
        super(GSA, self).__init__()
        self.normal = nn.BatchNorm2d(1)

    def forward(self, x):
        b, c, w, h = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x_g = torch.mean(x, dim=1, keepdim=True) + torch.max(x, dim=1, keepdim=True).values                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
        z_g_out = None
        for i in range(b):
            x_g_i = x_g[i].view((w * h))
            _, index = torch.topk(x_g_i, 1)
            r_index = int(index[0] / w)
            c_index = int(index[0] - w * r_index)
            mean = torch.tensor([1.0 * c_index, 1.0 * r_index])  # 均值
            covariance = torch.tensor([[1.0 * (w / 2) ** 2, 0.0], [0.0, 1.0 * (h / 2) ** 2]])  # 协方差矩阵                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
            mean = mean.cuda()
            covariance = covariance.cuda()
            gaussian_distribution = torch.distributions.multivariate_normal.MultivariateNormal(mean, covariance)                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
            x_g_index = torch.linspace(0, w, w).cuda()
            y_g_index = torch.linspace(0, h, h).cuda()
            x_g_index, y_g_index = torch.meshgrid(x_g_index, y_g_index)
            x_g_index = x_g_index.cuda()
            y_g_index = y_g_index.cuda()
            x_y_g_index = torch.stack([x_g_index.flatten(), y_g_index.flatten()], dim=1)                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!
            x_y_g_index = x_y_g_index.cuda()
            z_g = gaussian_distribution.log_prob(x_y_g_index)
            z_g = z_g.exp().reshape(x_g_index.shape)
            z_g = z_g.view([1, 1, w, h])
            if z_g_out is None:
                z_g_out = z_g
            else:
                z_g_out = torch.cat((z_g_out, z_g), dim=0)
        z_g_out = self.normal(z_g_out)
        return x * z_g_out

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 输入特征图 
    input_tensor = torch.randn(2, 32, 256, 256).to(device)    # (batch_size, channels, height, width)                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, 独家整理!

    model = GSA(32).to(device)

    print(model)

    # 前向传播
    output_tensor = model(input_tensor)

    # 打印输入输出形状
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")