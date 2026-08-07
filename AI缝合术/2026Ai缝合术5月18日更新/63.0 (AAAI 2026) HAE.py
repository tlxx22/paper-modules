import torch
import torch.nn as nn
import torch.nn.functional as F

class HAE(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1, proj_drop=0.,
        **kwargs, ):
        super().__init__()
        
        # maxpool占1/2的输入，attention占1/4的输入，dwconv占1/4的输入
        self.maxpool_in = maxpool_in = dim // 2 # 1/2 of original input (since dim is already X/2)
        self.attention_in = attention_in = dim // 4  # 1/8 of original input
        self.dwconv_in = dwconv_in = dim // 4  # 1/8 of original input
        
        self.maxpool_dim = maxpool_dim = maxpool_in * 2
        self.attention_dim = attention_dim = attention_in * 2
        self.dwconv_dim = dwconv_dim = dwconv_in * 2

        # MaxPool分支
        self.Maxpool = nn.MaxPool2d(kernel_size, stride=stride, padding=padding)
        self.proj_maxpool = nn.Conv2d(maxpool_in, maxpool_dim, kernel_size=1, stride=1, padding=0)
        self.mid_gelu_maxpool = nn.GELU()

        # Channel + Spatial Attention分支
        self.attention_conv = nn.Conv2d(attention_in, attention_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.attention_proj = nn.Conv2d(attention_dim, attention_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=attention_dim)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.mid_gelu_attention = nn.GELU()
        
        # 集成通道注意力和空间注意力
        self.channel_attention = self._build_channel_attention(attention_dim)
        self.spatial_attention = self._build_spatial_attention()

        # DWconv分支
        self.dwconv_conv = nn.Conv2d(dwconv_in, dwconv_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.dwconv_proj = nn.Conv2d(dwconv_dim, dwconv_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, groups=dwconv_dim)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.mid_gelu_dwconv = nn.GELU()

        # 融合层
        total_dim = maxpool_dim + attention_dim + dwconv_dim
        self.conv_fuse = nn.Conv2d(total_dim, total_dim, kernel_size=3, stride=1, padding=1, bias=False, groups=total_dim)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        self.proj = nn.Conv2d(total_dim, dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_channel_attention(self, in_planes, ratio=16):
        """构建通道注意力模块"""
        class ChannelAttentionModule(nn.Module):
            def __init__(self, in_planes, ratio):
                super().__init__()
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.max_pool = nn.AdaptiveMaxPool2d(1)
                self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
                self.relu1 = nn.ReLU()
                self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                # 输入已经是 B C H W 格式
                avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
                max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
                out = avg_out + max_out
                return self.sigmoid(out)
        
        return ChannelAttentionModule(in_planes, ratio)

    def _build_spatial_attention(self, kernel_size=7):
        """构建空间注意力模块"""
        class SpatialAttentionModule(nn.Module):
            def __init__(self, kernel_size):
                super().__init__()
                assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
                padding = 3 if kernel_size == 7 else 1
                self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                # 输入已经是 B C H W 格式
                avg_out = torch.mean(x, dim=1, keepdim=True)
                max_out, _ = torch.max(x, dim=1, keepdim=True)
                x = torch.cat([avg_out, max_out], dim=1)
                x = self.conv1(x)
                return self.sigmoid(x)
        
        return SpatialAttentionModule(kernel_size)
        
    def forward(self, x):

        # MaxPool分支 - 使用全部输入
        maxpool_x = x[:, :self.maxpool_in, :, :].contiguous()  # 使用全部输入 (1/2 of original)
        maxpool_x = self.Maxpool(maxpool_x)
        maxpool_x = self.proj_maxpool(maxpool_x)
        maxpool_x = self.mid_gelu_maxpool(maxpool_x)
        
        # Channel + Spatial Attention分支 - 使用1/4输入  
        attention_x = x[:, self.maxpool_in:self.maxpool_in+self.attention_in, :, :].contiguous()  # 取前1/4通道                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        attention_x = self.attention_conv(attention_x)
        attention_x = self.attention_proj(attention_x)
        attention_x = self.mid_gelu_attention(attention_x)
        
        # 应用通道注意力
        ca_weight = self.channel_attention(attention_x)
        attention_x = attention_x * ca_weight
        
        # 应用空间注意力
        sa_weight = self.spatial_attention(attention_x)
        attention_x = attention_x * sa_weight
        # DWconv分支 - 使用1/4输入
        dwconv_x = x[:, self.attention_in+self.maxpool_in:, :, :].contiguous()  # 取接下来1/4通道                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        dwconv_x = self.dwconv_conv(dwconv_x)
        dwconv_x = self.dwconv_proj(dwconv_x)
        dwconv_x = self.mid_gelu_dwconv(dwconv_x)

        # 拼接三个分支
        x = torch.cat((maxpool_x, attention_x, dwconv_x), dim=1)

        # 融合和投影
        x = x + self.conv_fuse(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 64, 32, 32).to(device)                                                                                                                                                                                            # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

    model = HAE(dim=64).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")