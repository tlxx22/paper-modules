import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

"""
    论文地址：https://openaccess.thecvf.com/content/CVPR2026/papers/Cogo_Leveraging_Multispectral_Sensors_for_Color_Correction_in_Mobile_Cameras_CVPR_2026_paper.pdf
    论文题目：Leveraging Multispectral Sensors for Color Correction in Mobile Cameras (CVPR 2026)
    中文题目：利用多光谱传感器实现移动相机的色彩校正(CVPR 2026)
    讲解视频：https://www.bilibili.com/video/BV19TET6xEoy/
    多尺度颜色注意力（Multi-Scale Color Attention ,MCA）
        实际意义：①颜色通道之间相互独立建模的问题：传统颜色校正方法常把RGB三通道近似看成可以分别处理，但真实颜色偏差并不是每个通道单独变化，而是R、G、B 通道之间存在耦合关系。
                ②不同尺度色彩信息利用不充分：色彩偏差同时出现在全局光照与局部纹理，单一尺度特征易丢失细节。
                ③复杂光照下色彩校正鲁棒性差：混合光照、强色偏场景易出现局部色偏、校正过度/不足的问题。
        实现方式：通过多尺度卷积生成 Q/K/V 颜色特征，并利用注意力机制建模通道间依赖关系，从而实现更精细、更稳定的非线性颜色校正。
"""

class MultiScaleColorAttention(nn.Module):
    # 初始化函数：定义模块所有可学习参数、网络层
    # dim：输入特征图的通道数  num_heads：多头注意力的头数  bias：卷积层是否启用偏置项
    def __init__(self, dim, num_heads, bias=False):
        # 调用父类nn.Module的初始化方法，必写语句
        super(MultiScaleColorAttention, self).__init__()

        # 保存多头注意力的头数，类内全局使用
        self.num_heads = num_heads

        # ========== 可学习温度系数（注意力缩放参数）==========
        # Query与Anchor之间注意力的温度参数，可训练，用于缩放注意力分数
        self.query_anchor_temperature = nn.Parameter(
            torch.ones(num_heads, 1, 1)
        )
        # Anchor与Key之间注意力的温度参数，可训练，控制注意力分布平滑度
        self.anchor_key_temperature = nn.Parameter(
            torch.ones(num_heads, 1, 1)
        )

        # ========== Query特征提取分支 ==========
        # 3*3卷积，步长2下采样、反射填充，分组卷积（分组数=通道数=深度卷积），提取Query特征
        self.query_embed = nn.Conv2d(
            dim,                # 输入通道数
            dim,                # 输出通道数
            kernel_size=3,      # 卷积核大小3×3
            stride=2,           # 步长2，实现特征图空间尺寸下采样
            padding=1,          # 填充1个像素，保证卷积后尺寸匹配
            padding_mode='reflect', # 反射填充，避免图像边缘伪影（色彩任务常用）
            groups=dim,         # 分组卷积，每个通道单独卷积，降低参数量
            bias=bias           # 是否使用偏置
        )

        # ========== Key特征提取分支 ==========
        # 3*3普通卷积，步长2下采样，提取Key特征（注意力机制的匹配键）
        self.key_embed = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode='reflect',
            bias=bias
        )

        # ========== Value特征提取分支 ==========
        # 1*1逐点卷积，仅做通道变换，不改变空间尺寸，提取Value特征（注意力聚合的值）
        self.value_embed = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=bias
        )

        # ========== Anchor锚点特征提取分支（论文核心设计） ==========
        # 串行两层卷积，先深度卷积下采样，再1*1卷积压缩通道，生成锚点特征
        self.anchor_embed = nn.Sequential(
            # 第一层：3×3深度卷积，步长2下采样
            nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode='reflect',
                groups=dim,
                bias=bias
            ),
            # 第二层：1×1卷积，将通道数减半，压缩特征维度
            nn.Conv2d(
                dim,
                dim // 2,
                kernel_size=1
            )
        )

        # ========== 输出投影层 ==========
        # 1*1卷积，对注意力聚合后的特征做通道融合，输出最终增强特征
        self.output_projection = nn.Conv2d(
            dim,
            dim,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x):
        # 获取输入特征图的维度：b=批次 c=通道 h=高度 w=宽度
        b, c, h, w = x.shape

        # ==============================================
        # 第一步：四个分支分别提取 Query / Key / Value / Anchor 特征
        # ==============================================
        # 输入特征经过卷积，得到Query特征
        query_feat = self.query_embed(x)
        # 输入特征经过卷积，得到Key特征
        key_feat = self.key_embed(x)
        # 先卷积得到Value基础特征，再与原始输入逐元素相乘，融合原始信息
        value_feat = self.value_embed(x) * x
        # 经过串行卷积，得到锚点Anchor特征
        anchor_feat = self.anchor_embed(x)

        # ==============================================
        # 第二步：张量维度重排，适配多头注意力计算【多头注意力计算】
        # 把通道维度拆分为【多头数 + 单头通道】，空间维度展平为一维序列
        # einops.rearrange：灵活变换维度，格式：原维度 -> 新维度
        # ==============================================
        # Query维度重排：(b, head*c, h, w) → (b, head, c, h*w)
        query_feat = rearrange(query_feat,'b (head c) h w -> b head c (h w)',head=self.num_heads)
        # Key维度重排，格式与Query保持一致
        key_feat = rearrange(key_feat,'b (head c) h w -> b head c (h w)',head=self.num_heads)
        # Value维度重排，统一多头格式
        value_feat = rearrange(value_feat,'b (head c) h w -> b head c (h w)',head=self.num_heads)
        # Anchor维度重排，适配后续注意力矩阵计算
        anchor_feat = rearrange(anchor_feat,'b (head c) h w -> b head c (h w)',head=self.num_heads)

        # ==============================================
        # 第三步：特征L2归一化，提升注意力匹配稳定性（色彩任务常用）
        # dim=-1：在最后一维（空间序列维度）做归一化
        # ==============================================
        query_feat = F.normalize(query_feat, dim=-1)
        key_feat = F.normalize(key_feat, dim=-1)
        anchor_feat = F.normalize(anchor_feat, dim=-1)

        # ==============================================
        # 第四步：第一阶段注意力：Query ↔ Anchor 注意力计算
        # 矩阵相乘 = 特征相似度计算，乘以温度系数缩放分数，最后Softmax归一化得到注意力权重
        # ==============================================
        query_anchor_attn = (query_feat @ anchor_feat.transpose(-2, -1)) * self.query_anchor_temperature
        # Softmax将相似度转为0~1之间的概率分布（注意力权重）
        query_anchor_attn = query_anchor_attn.softmax(dim=-1)

        # ==============================================
        # 第五步：第二阶段注意力：Anchor ↔ Key 注意力计算
        # 双层级联注意力（Query→Anchor→Key），是本模块多尺度设计核心
        # ==============================================
        anchor_key_attn = (anchor_feat @ key_feat.transpose(-2, -1)) * self.anchor_key_temperature
        # 归一化得到注意力权重
        anchor_key_attn = anchor_key_attn.softmax(dim=-1)

        # ==============================================
        # 第六步：注意力加权聚合Value特征
        # 先用Anchor-Key注意力加权Value，再用Query-Anchor注意力二次加权，得到增强特征
        # ==============================================
        # 第一层聚合：注意力权重 × Value特征
        anchor_value_feat = anchor_key_attn @ value_feat
        # 第二层聚合：二次注意力加权，得到最终增强特征
        enhanced_feature = query_anchor_attn @ anchor_value_feat

        # ==============================================
        # 第七步：恢复原始空间维度
        # 将展平的空间序列(h*w)还原为二维特征图(h, w)，合并多头通道
        # ==============================================
        enhanced_feature = rearrange(
            enhanced_feature,
            'b head c (h w) -> b (head c) h w',
            head=self.num_heads,
            h=h,  # 还原原始特征图高度
            w=w   # 还原原始特征图宽度
        )

        # ==============================================
        # 第八步：输出投影，通道融合得到最终结果
        # ==============================================
        enhanced_feature = self.output_projection(enhanced_feature)
        # 返回经过注意力增强的特征图
        return enhanced_feature

if __name__ == "__main__":
    input_feature = torch.randn(1, 64, 128, 128)
    model = MultiScaleColorAttention(dim=64, num_heads=8, bias=False)
    output_feature = model(input_feature)
    print("输入特征维度：", input_feature.shape)
    print("输出特征维度：", output_feature.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")