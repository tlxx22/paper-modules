import torch
import torch.nn as nn
import torch.nn.functional as F

class AlternateCat(nn.Module):
    def __init__(self, dim=1, num=3):
        """
        沿指定维度交替拼接两个张量。

        Args:
            dim (int, optional): 指定的维度，默认为 1（通道维度）。
        """
        super().__init__()
        self.dim = dim
        self.num = num

    def forward(self, x_list):
        """
        沿指定维度交替拼接两个张量。

        Args:
            x (torch.Tensor): 第一个输入张量。
            y (torch.Tensor): 第二个输入张量。

        Returns:
            torch.Tensor: 沿指定维度交替拼接后的张量。

        Raises:
            AssertionError: 如果输入张量在指定维度上的大小不一致。
        """
        # 确保两个张量在指定维度上的大小一致
        # assert x.shape == y.shape, f'x.shape:{x.shape} != y.shape:{y.shape}'
        assert len(x_list) == self.num, f'input num error!'
        for i in range(self.num):
            assert x_list[0].shape == x_list[i].shape, f'input index{i} shape error!'



        # 获取指定维度的大小
        size = x_list[0].size(self.dim)
        # print(size)

        x_list_slices = []
        # 将 x 和 y 沿着指定维度拆分为单个元素的切片
        for i in range(self.num):
            x_list_slices.append(torch.split(x_list[i], 1, dim=self.dim))

        # 交替拼接 x 和 y 的切片
        interleaved_slices = []
        for i in range(size):
            for j in range(self.num):
                interleaved_slices.append(x_list_slices[j][i])
            # interleaved_slices.extend([x_slices[i], y_slices[i]])

        # 沿着指定维度堆叠交替后的切片
        concatenated = torch.cat(interleaved_slices, dim=self.dim)

        return concatenated
    
class AlCattention(nn.Module):
    def __init__(self, dim):
        super(AlCattention, self).__init__()
        self.dim = dim  # 保存通道维度参数
        # 自适应平均池化：将空间维度压缩为1x1（保留通道维度信息）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 自适应最大池化：同样压缩空间维度，捕捉通道维度的最大值信息
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # MLP多层感知机：用于将统计特征映射为通道权重
        self.alcat = AlternateCat(dim=1, num=3)
        self.share_mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=1 , stride=1, padding=0, bias=True, groups=self.dim),                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1 , stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.spconv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3 , stride=1, padding=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        std = torch.std(x, dim=(2, 3), keepdim=True)  # 空间维度标准差：(B, 2C)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        x_ams = self.alcat([x_avg, x_max, std])
        channel_weights = self.share_mlp(x_ams)
        x_1 = x * channel_weights + x

        avg_out = torch.mean(x_1, dim=1, keepdim=True)
        max_out, _ = torch.max(x_1, dim=1, keepdim=True)

        # 空间特征拼接
        spatial_features = torch.cat([avg_out,  max_out], dim=1)
        spatial_weights = self.spconv(spatial_features)
        dwt = x_1 * spatial_weights
        return dwt

# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 64, 128, 128).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = AlCattention(dim=64).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")