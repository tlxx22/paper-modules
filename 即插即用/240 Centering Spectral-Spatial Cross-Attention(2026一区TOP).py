import torch
from torch import nn

"""
    论文地址：https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11359294
    论文题目：Beyond Degradation Redundancy: Contrastive Prompt Learning for All-in-One Image Restoration (2026一区TOP)
    中文题目：面向高光谱图像分类的光谱选择卷积与重叠中心 Mamba 网络(2026一区TOP)
    讲解视频：https://www.bilibili.com/video/BV16gG96EEpd/
    中心光谱-空间交叉注意力（Centering Spectral-Spatial Cross-Attention,CSSCA）
        实际意义：①光谱特征与空间特征相互割裂的问题：高光谱图像分类既需要利用中心像素的光谱信息，也需要利用其周围区域的空间结构信息。然而，若仅在不同分支中分别提取光谱特征和空间特征，二者之间缺少显式交互：
                    中心像素的光谱判别信息无法有效指导空间区域筛选；周围空间结构也无法反向增强中心像素类别表达。
                ②中心像素先验利用不足的问题：普通特征提取过程通常会将中心像素与周围像素同等处理，导致中心像素的重要光谱响应被邻域背景稀释；在边界区域或混合地物区域中，邻近类别对中心像素产生干扰；
                    少样本条件下，模型难以稳定提取与类别最相关的特征。
        实现方式：通过中心像素特征与全局空间特征的交叉建模，同时生成空间注意力与增强中心先验，实现多分支的有效协同。
"""

class SpatialCrossAttention(nn.Module):
    """
    空间交叉注意力模块
    功能：
        1. 分别提取高度方向和宽度方向的空间上下文信息；
        2. 利用全局池化信息增强局部空间特征；
        3. 融合两条分支产生最终空间注意力图；
        4. 使用空间注意力图重新加权输入特征。
    """
    def __init__(self, feature_channels):
        super(SpatialCrossAttention, self).__init__()

        # 对通道维响应进行归一化，生成通道权重
        self.channel_softmax = nn.Softmax(dim=-1)

        # 提取整幅特征图的全局上下文信息
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 压缩宽度维，保留高度方向的空间上下文
        self.height_context_pool = nn.AdaptiveAvgPool2d((None, 1))

        # 压缩高度维，保留宽度方向的空间上下文
        self.width_context_pool = nn.AdaptiveAvgPool2d((1, None))

        # 对每个通道独立进行归一化
        self.channel_group_norm = nn.GroupNorm(
            num_groups=feature_channels,
            num_channels=feature_channels
        )

        # 融合高度方向与宽度方向的上下文信息
        self.direction_fusion_conv = nn.Conv2d(
            in_channels=feature_channels,
            out_channels=feature_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        # 提取局部空间特征
        self.local_feature_conv = nn.Conv2d(
            in_channels=feature_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, input_feature):
        # 读取输入特征的形状信息
        batch_size, channel_count, height, width = input_feature.size()

        # 当前模块处理的是一个完整特征组，形状保持不变
        group_feature = input_feature.reshape(
            batch_size, channel_count, height, width
        )

        # ==========================================================
        # 分支一：方向空间上下文增强分支
        # ==========================================================
        # 沿宽度方向压缩，得到高度方向上下文：[B, C, H, 1]
        height_context = self.height_context_pool(group_feature)
        # 沿高度方向压缩，得到宽度方向上下文：[B, C, 1, W]
        # 转置后变为 [B, C, W, 1]，便于与高度方向特征拼接
        width_context = self.width_context_pool(group_feature).permute(0, 1, 3, 2)
        # 在空间维上拼接两个方向的上下文信息：[B, C, H+W, 1]
        directional_context = torch.cat([height_context, width_context],dim=2)
        # 使用 1×1 卷积融合高度方向与宽度方向信息
        directional_context = self.direction_fusion_conv(directional_context)

        # 将融合后的上下文重新拆分为高度注意力与宽度注意力
        height_attention, width_attention = torch.split(directional_context,[height, width],dim=2)
        # 将宽度注意力恢复到 [B, C, 1, W] 的排列形式
        width_attention = width_attention.permute(0, 1, 3, 2)
        # 分别生成高度方向与宽度方向门控，并增强原始特征
        direction_enhanced_feature = self.channel_group_norm(group_feature* height_attention.sigmoid()* width_attention.sigmoid())
        # 从方向增强特征中提取通道权重：[B, 1, C]
        direction_channel_weight = self.channel_softmax(self.global_avg_pool(direction_enhanced_feature).reshape(batch_size, channel_count, 1).permute(0, 2, 1))
        # 将方向增强特征展平为空间序列：[B, C, H×W]
        direction_spatial_feature = direction_enhanced_feature.reshape(batch_size, channel_count, -1)

        # ==========================================================
        # 分支二：全局先验增强分支
        # ==========================================================
        # 使用 3×3 卷积提取局部空间特征
        local_spatial_feature = self.local_feature_conv(group_feature)
        # 对输入特征进行归一化和门控，获得基础响应图
        normalized_gate_feature = self.channel_group_norm(group_feature).sigmoid()
        # 从输入特征中提取全局通道先验信息
        global_prior_feature = self.global_avg_pool(self.channel_group_norm(group_feature)).sigmoid()
        # 使用全局先验调制门控特征，并注入到局部空间特征中
        global_prior_enhanced_feature = (local_spatial_feature+ global_prior_feature * normalized_gate_feature)
        # 从全局先验增强特征中提取通道权重：[B, 1, C]
        global_prior_channel_weight = self.channel_softmax(self.global_avg_pool(global_prior_enhanced_feature).reshape(batch_size, channel_count, 1).permute(0, 2, 1))
        # 全局特征
        cross_fusion_spatial_feature = global_prior_channel_weight.reshape(batch_size, channel_count, -1)

        # ==========================================================
        # 双分支交叉融合
        # ==========================================================
        # 方向分支生成的空间响应：[B, 1, H×W]
        direction_response = torch.matmul(direction_channel_weight,direction_spatial_feature)
        # 全局先验分支生成的空间响应：[B, 1, H×W]
        global_prior_response = torch.matmul(global_prior_channel_weight,cross_fusion_spatial_feature)
        # 融合两条分支的响应，恢复为空间注意力图：[B, 1, H, W]
        spatial_attention_map = (direction_response + global_prior_response).reshape(batch_size, 1, height, width)
        # 使用最终空间注意力图对当前组特征进行逐位置增强
        output_feature = group_feature * spatial_attention_map.sigmoid()
        return output_feature.reshape(batch_size, channel_count, height, width)

class CascadedSpatialCrossAttention(nn.Module):
    """
    级联空间交叉注意力模块

    功能：
    1. 将输入特征沿通道维划分为多个特征组；
    2. 每组分别经过一个独立的空间交叉注意力模块；
    3. 当前组输入与前一组输出相加，实现组间级联信息传递；
    4. 将各组输出重新拼接为完整特征图。
    """

    def __init__(self, feature_channels, num_groups):
        super(CascadedSpatialCrossAttention, self).__init__()

        # 检查通道数能否被分组数量整除
        if feature_channels % num_groups != 0:
            raise ValueError(
                f"feature_channels={feature_channels} 必须能够被 "
                f"num_groups={num_groups} 整除。"
            )

        # 保存特征分组数量
        self.num_groups = num_groups

        # 计算每个特征组包含的通道数
        channels_per_group = feature_channels // num_groups

        # 为每个通道组构建独立的空间交叉注意力模块
        self.group_attention_blocks = nn.ModuleList(
            [
                SpatialCrossAttention(channels_per_group)
                for _ in range(num_groups)
            ]
        )

    def forward(self, input_feature):
        # 按通道维将输入特征切分为多个特征组
        group_input_features = input_feature.chunk(self.num_groups,dim=1)
        # 用于保存每一组经过增强后的输出特征
        group_output_features = []

        # 初始化第一个待处理的特征组
        current_group_feature = group_input_features[0]
        # 逐组进行级联空间注意力增强
        for group_index in range(self.num_groups):
            # 从第二组开始，将前一组输出累加到当前组输入中
            if group_index > 0:
                current_group_feature = (current_group_feature+ group_input_features[group_index])
            # 对当前组执行空间交叉注意力增强
            current_group_feature = self.group_attention_blocks[group_index](current_group_feature)
            # 保存当前组输出
            group_output_features.append(current_group_feature)
        # 在通道维拼接所有组的输出，恢复完整特征图
        output_feature = torch.cat(group_output_features, dim=1)
        # 返回级联增强后的输出特征
        return output_feature

if __name__ == "__main__":
    # 构造一个示例输入特征：[批大小, 通道数, 高度, 宽度]
    input_feature = torch.randn(1, 64, 32, 32)
    # 构建级联空间交叉注意力模块，将 64 个通道划分为 4 组
    model = CascadedSpatialCrossAttention(feature_channels=64,num_groups=4)
    output_feature = model(input_feature)
    print("输入特征维度：", input_feature.shape)
    print("输出特征维度：", output_feature.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")