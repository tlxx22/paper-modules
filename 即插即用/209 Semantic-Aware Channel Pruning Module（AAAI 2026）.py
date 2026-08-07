import torch                              # 导入 PyTorch 主库
import torch.nn as nn                     # 导入 PyTorch 神经网络模块
import torch.nn.functional as F           # 导入常用函数式接口（插值、激活等）
from torchvision.models import convnext_base  # 导入 ConvNeXt-Base 主干网络（用于提供额外通道权重先验）

"""
    论文地址：https://arxiv.org/pdf/2511.12432
    论文题目：Text-Guided Channel Perturbation and Pretrained Knowledge Integration for Unified Multi-Modality Image Fusion（AAAI 2026）
    中文题目：面向统一多模态图像融合的文本引导通道扰动与预训练知识融合方法（AAAI 2026）
    讲解视频：https://www.bilibili.com/video/BV1LoA8zqEZ8/
    语义感知通道剪枝模块（Semantic-Aware Channel Pruning Module，SCPM）
        实际意义：①多模态数据的固有冗余与特征干扰问题：多模态图像（如红外-可见光）特征包含重复、无效的噪声信息，冗余比例高。如果不加约束，会引入无效特征，降低关键结构与语义关注。
                ②缺乏全局语义指导的通道筛选问题：仅从局部特征响应角度判断通道重要性，缺乏全局语义层面的指导，在面对跨模态、跨任务的融合场景时，容易错误选择通道，导致模型泛化能力不足。
        实现方式：①通道注意力机制生成通道权重，衡量各通道重要性。
                ②利用预训练视觉模型（如 ConvNeXt）提取融合特征的高层语义表示。
                ③权重融合与通道筛选融合两类权重，根据融合权重进行 Top-k 通道选择（保留约 70% 通道），删除冗余的低贡献通道。
                ④通过 1×1 卷积恢复通道维度，供后续模块使用。
"""
class Semantic_Aware_Channel_Pruning_Module(nn.Module):
    def __init__(
        self,
        in_channels,
        reduction_ratio=8,
        keep_ratio=0.7,
        prior_fusion_init=1.0,
        use_convnext_prior=True
    ):
        super().__init__()

        # 输入特征图的通道数 C
        self.in_channels = in_channels

        # 计算需要保留的通道数量（Top-K 通道数）
        # 例如 keep_ratio=0.7 表示保留 70% 通道
        self.topk_channels = int(in_channels * keep_ratio)

        # 是否启用 ConvNeXt 语义先验分支
        self.use_convnext_prior = use_convnext_prior

        # ==============================
        # ① 通道全局描述子（Global Descriptor）
        # ==============================
        # 使用自适应平均池化，将每个通道压缩为 1×1
        # 输出形状：B × C × 1 × 1
        self.global_descriptor = nn.AdaptiveAvgPool2d(1)

        # ==============================
        # ② 通道重要性预测器（SE 风格 MLP）
        # ==============================
        # 这是一个两层全连接网络：
        # C → C/reduction → C
        # 用来预测每个通道的重要性权重
        self.channel_importance_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),  # 降维
            nn.ReLU(inplace=True),                                    # 非线性
            nn.Linear(in_channels // reduction_ratio, in_channels),   # 升维回原始通道数
            nn.Sigmoid()                                               # 输出0~1之间的权重
        )

        # ==============================
        # ③ 可选的 ConvNeXt 语义先验分支
        # ==============================
        if use_convnext_prior:

            # 加载预训练 ConvNeXt-Base 网络
            backbone = convnext_base(pretrained=True)

            # 去掉最后的分类层，只保留特征提取部分
            self.prior_backbone = nn.Sequential(*list(backbone.children())[:-2])

            # 将 ConvNeXt 输出的 1024 维特征映射到 C 维
            self.prior_mapper = nn.Linear(1024, in_channels)
        else:
            self.prior_backbone = None
            self.prior_mapper = None

        # ==============================
        # ④ 输入特征 → RGB 映射
        # ==============================
        # 将输入的 C 通道特征映射成 3 通道
        # 目的是喂给 ConvNeXt（它接受RGB图像）
        self.rgb_projection = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)

        # ==============================
        # ⑤ 剪枝后通道重投影层
        # ==============================
        # 剪枝后只剩 topk_channels
        # 用 1×1 卷积恢复回原始 C 通道
        self.channel_reprojection = nn.Conv2d(
            self.topk_channels, in_channels, kernel_size=1, bias=False
        )

        # ==============================
        # ⑥ 可学习的语义先验融合权重
        # ==============================
        # 控制 ConvNeXt 先验影响强度
        self.prior_fusion_weight = nn.Parameter(
            torch.tensor(prior_fusion_init), requires_grad=True
        )

    def forward(self, x):
        # 输入尺寸：B × C × H × W
        B, C, H, W = x.size()

        # ==================================================
        # Step 1：计算基础通道重要性（SE 机制）
        # ==================================================
        # 全局平均池化 → B × C
        pooled = self.global_descriptor(x).view(B, -1)
        # 通过 MLP 预测每个通道的重要性分数
        base_importance = self.channel_importance_mlp(pooled)

        # ==================================================
        # Step 2：注入 ConvNeXt 语义先验（可选）
        # ==================================================
        if self.use_convnext_prior and self.prior_backbone is not None:
            # 先把输入映射为3通道
            rgb_feat = self.rgb_projection(x)
            # Resize 到 224×224（ConvNeXt标准输入尺寸）
            resized_feat = F.interpolate(
                rgb_feat,
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )

            # 冻结 ConvNeXt 权重，仅提取特征
            with torch.no_grad():
                prior_feat = self.prior_backbone(resized_feat)
                # 拉平成 B × 1024 × HW
                # 再做空间平均
                prior_feat = prior_feat.flatten(2).mean(dim=2)

            # 映射到 C 维，并经过 Sigmoid
            prior_importance = torch.sigmoid(
                self.prior_mapper(prior_feat)
            )

            # 融合基础重要性和语义先验 通道权重 + 可学习的参数 ✖ ConvNeXt骨干处理后的权重
            importance_scores = (
                base_importance +
                self.prior_fusion_weight * prior_importance
            )
        else:
            # 如果没有先验，直接使用基础重要性
            importance_scores = base_importance

        # ==================================================
        # Step 3：Top-K 通道选择
        # ==================================================
        # 对每个样本选出重要性最高的 K 个通道
        _, topk_indices = torch.topk(
            importance_scores,
            self.topk_channels,
            dim=1
        )
        # 扩展索引维度，用于从特征图中提取通道
        topk_indices = topk_indices.unsqueeze(-1).unsqueeze(-1)
        topk_indices = topk_indices.expand(-1, -1, H, W)
        # 按通道维度 gather，得到剪枝后的特征
        pruned_feature = x.gather(1, topk_indices)

        # ==================================================
        # Step 4：通道重建
        # ==================================================
        # 将 K 通道重新映射回原始 C 通道
        reconstructed_feature = self.channel_reprojection(pruned_feature)
        return reconstructed_feature

if __name__ == "__main__":
    x = torch.rand(1, 32, 50, 50)
    model = Semantic_Aware_Channel_Pruning_Module(32)
    output = model(x)
    print(f"输入张量X形状: {x.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")