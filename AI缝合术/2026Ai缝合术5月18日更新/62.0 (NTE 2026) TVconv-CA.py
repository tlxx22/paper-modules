import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class _ConvBlock(nn.Sequential):
    """
    _ConvBlock类定义了一个简单的卷积块，包含卷积层、层归一化和ReLU激活函数。
    """
    def __init__(self, in_planes, out_planes, h, w, kernel_size=3, stride=1, bias=False):
        # 计算填充大小，使得输出大小与输入大小相同
        padding = (kernel_size - 1) // 2
        super(_ConvBlock, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=bias),  # 卷积层
            nn.LayerNorm([out_planes, h, w]),  # 层归一化
            nn.ReLU(inplace=True)  # ReLU激活函数
        )


class TVConv(nn.Module):
    """
    TVConv类定义了一个基于位置映射的空间变体卷积模块。
    """
    def __init__(self,
                 channels,
                 TVConv_k=3,
                 stride=1,
                 TVConv_posi_chans=4,
                 TVConv_inter_chans=64,
                 TVConv_inter_layers=3,
                 TVConv_Bias=False,
                 h=32,  # 修改默认值为32
                 w=32,  # 修改默认值为32
                 **kwargs):
        super(TVConv, self).__init__()
        self.channels = channels
        self.TVConv_k = TVConv_k
        self.h = h  # 添加这行，保存h为实例属性
        self.w = w  # 添加这行，保存w为实例属性

        self.bias_layers = None
        self.TVConv_k_square = TVConv_k * TVConv_k

        # 计算输出通道数
        out_chans = self.TVConv_k_square * self.channels

        # 初始化位置映射参数
        self.posi_map = nn.Parameter(torch.Tensor(1, TVConv_posi_chans, h, w))
        nn.init.ones_(self.posi_map)  # 用1初始化

        # 创建权重层和偏置层
        self.weight_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, out_chans, TVConv_inter_layers, h, w)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        if TVConv_Bias:
            self.bias_layers = self._make_layers(TVConv_posi_chans, TVConv_inter_chans, channels, TVConv_inter_layers, h, w)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        # 初始化 Unfold 模块，用于提取局部区域
        self.unfold = nn.Unfold(TVConv_k, 1, (TVConv_k-1)//2, stride)

    def _make_layers(self, in_chans, inter_chans, out_chans, num_inter_layers, h, w):
        """
        创建卷积层序列。
        """
        layers = [_ConvBlock(in_chans, inter_chans, h, w, bias=False)]
        for i in range(num_inter_layers):
            layers.append(_ConvBlock(inter_chans, inter_chans, h, w, bias=False))
        layers.append(nn.Conv2d(
            in_channels=inter_chans,
            out_channels=out_chans,
            kernel_size=3,
            padding=1,
            bias=False))  # 最后一层卷积
        return nn.Sequential(*layers)

    def forward(self, x):
        # 计算卷积权重
        weight = self.weight_layers(self.posi_map)
        weight = weight.view(1, self.channels, self.TVConv_k_square, self.h, self.w) # torch.Size([1, 64, 9, 32, 32])
        # 利用 Unfold 模块获取局部区域，并按照权重进行加权求和
        out = self.unfold(x).view(x.shape[0], self.channels, self.TVConv_k_square, self.h, self.w) # torch.Size([2, 64, 9, 32, 32])
        """
            weight * out：对这两个张量在 TVConv_k_square 维度上进行逐元素相乘。这个操作相当于对每个位置的局部区域应用一个位置特定的卷积核。
            .sum(dim=2) ：在TVConv_k_square维度上对乘积结果进行求和。TVConv_k_square 代表卷积核的展开大小（即核的面积），
                所以这个求和操作相当于对每个局部区域的卷积结果进行加权求和，类似于传统卷积操作。        
        """
        out = (weight * out).sum(dim=2) #实现了基于位置的加权卷积操作，生成了一个新的特征图。 # torch.Size([2, 64, 32, 32])
        if self.bias_layers is not None:
            # 如果使用偏置，则加上偏置
            bias = self.bias_layers(self.posi_map)
            out = out + bias
        return out


def get_freq_indices(method):
    # 确保方法在指定的选项中
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    # 从方法名中提取频率数
    num_freq = int(method[3:])
    if 'top' in method:
        # 预定义的 top 频率索引
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        # 选择前 num_freq 个索引
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        # 预定义的 low 频率索引
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        # 选择前 num_freq 个索引
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        # 预定义的 bot 频率索引
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        # 选择前 num_freq 个索引
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        # 如果方法不在选项中，抛出异常
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiFrequencyChannelAttention(nn.Module):
    def __init__(self,
                 in_channels,
                 dct_h=7,
                 dct_w=7,
                 frequency_branches=16,
                 frequency_selection='top',
                 reduction=16,
                 feature_h=32,  # 添加特征图高度参数
                 feature_w=32): # 添加特征图宽度参数
        super(MultiFrequencyChannelAttention, self).__init__()

        # 确保频率分支数是有效的
        assert frequency_branches in [1, 2, 4, 8, 16, 32]
        # 构造频率选择字符串
        frequency_selection = frequency_selection + str(frequency_branches)

        self.num_freq = frequency_branches
        self.dct_h = dct_h
        self.dct_w = dct_w

        # 获取频率索引
        mapper_x, mapper_y = get_freq_indices(frequency_selection)
        self.num_split = len(mapper_x)
        # 根据 DCT 大小调整索引
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # 确保 mapper_x 和 mapper_y 长度一致
        assert len(mapper_x) == len(mapper_y)

        # 初始化 DCT 权重
        for freq_idx in range(frequency_branches):
            self.register_buffer('dct_weight_{}'.format(freq_idx), self.get_dct_filter(dct_h, dct_w, mapper_x[freq_idx], mapper_y[freq_idx], in_channels))                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        # 定义全连接层
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, padding=0, bias=False))

        # 定义自适应池化层
        self.average_channel_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_channel_pooling = nn.AdaptiveMaxPool2d(1)
        # 修改：使用传入的in_channels和feature_h/feature_w
        self.model = TVConv(in_channels, h=feature_h, w=feature_w)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        # 初始化 DCT 滤波器
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        # 构建 DCT 滤波器
        for t_x in range(tile_size_x):
            for t_y in range(tile_size_y):
                dct_filter[:, t_x, t_y] = self.build_filter(t_x, mapper_x, tile_size_x) * self.build_filter(t_y, mapper_y, tile_size_y)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        return dct_filter

    def build_filter(self, pos, freq, POS):
        # 计算 DCT 滤波器值
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def forward(self, x):
        # 获取输入的形状
        batch_size, C, H, W = x.shape

        x_pooled = x
        # 如果输入大小与 DCT 大小不匹配，进行自适应池化
        if H != self.dct_h or W != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        # 初始化频谱特征
        multi_spectral_feature_avg, multi_spectral_feature_max, multi_spectral_feature_min = 0, 0, 0                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!
        for name, params in self.state_dict().items():
            # 循环遍历模型的状态字典，该字典包含模型的所有参数。它寻找名称中包含 'dct_weight' 的参数。
            if 'dct_weight' in name:
                # 计算频谱特征：将输入与 DCT 权重参数逐元素相乘
                x_pooled_spectral = x_pooled * params
                # 累加池化频谱特征的平均值
                multi_spectral_feature_avg += self.average_channel_pooling(x_pooled_spectral)
                # 累加池化频谱特征的最大值
                multi_spectral_feature_max += self.max_channel_pooling(x_pooled_spectral)
                # 累加池化频谱特征的最小值：通过取反后最大池化的方法实现
                multi_spectral_feature_min += -self.max_channel_pooling(-x_pooled_spectral)


        # 归一化频谱特征
        multi_spectral_feature_avg = multi_spectral_feature_avg / self.num_freq
        multi_spectral_feature_max = multi_spectral_feature_max / self.num_freq
        multi_spectral_feature_min = multi_spectral_feature_min / self.num_freq

        # 通过全连接层生成注意力图
        multi_spectral_avg_map = self.fc(multi_spectral_feature_avg).view(batch_size, C, 1, 1)
        multi_spectral_max_map = self.fc(multi_spectral_feature_max).view(batch_size, C, 1, 1)
        multi_spectral_min_map = self.fc(multi_spectral_feature_min).view(batch_size, C, 1, 1)

        # 计算最终的注意力图
        multi_spectral_attention_map = F.sigmoid(multi_spectral_avg_map + multi_spectral_max_map + multi_spectral_min_map)                                                                                                                                                                                             # 哔哩哔哩/微信公众号: A-I-缝-合-术, AI-Feng-he-shu, 缝-合-术-AI, AIf-eng-hes-hu独家整理!

        # 将注意力图应用于输入
        x = x * multi_spectral_attention_map.expand_as(x)
        x=self.model(x)
        return x


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(2, 16, 32, 32).to(device)
    # TVconv-CA
    model = MultiFrequencyChannelAttention(
        in_channels=16, 
        dct_h=32, 
        dct_w=32, 
        frequency_branches=16, 
        frequency_selection='top',
        feature_h=32,  # 传入特征图高度
        feature_w=32   # 传入特征图宽度
    ).to(device)
    
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")