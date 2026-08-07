import torch                              # 导入 PyTorch 主库
import torch.nn as nn                     # 导入 PyTorch 神经网络模块

"""
    论文地址：https://arxiv.org/pdf/2511.12432
    论文题目：Text-Guided Channel Perturbation and Pretrained Knowledge Integration for Unified Multi-Modality Image Fusion（AAAI 2026）
    中文题目：面向统一多模态图像融合的文本引导通道扰动与预训练知识融合方法（AAAI 2026）
    讲解视频：https://www.bilibili.com/video/BV1gbPPz9EQA/
    几何仿射调制模块（Geometric Affine Modulation Module，GAM）
        实际意义：①直接注入模态信息导致过拟合问题：在多模态图像融合中，传统多编码器直接将模态特征深度耦合，虽然模型融合效果可能更强，但容易过度依赖特定模态的统计特征，导致对训练特征的模态分布产生偏置，导致泛化能力下降。
        ②单编码器融合中模态区分能力不足问题：在融合过程中不同模态特征被过度混合，由此导致模态间差异被削弱，特定模态优势信息（如红外热目标、医学成像高对比结构）表达不充分。
        实现方式：GAM通过利用模态全局统计信息生成仿射调制参数（γ、β），以几何变换的方式调节融合特征分布，从而增强模态表达能力并抑制过拟合。
"""

class Geometric_Affine_Modulation(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # =========================================
        # ① 全局通道描述子提取器
        # =========================================
        # 对输入特征做全局平均池化
        # 输出尺寸：B × C × 1 × 1
        # 表示每个通道的全局统计信息
        self.global_descriptor = nn.AdaptiveAvgPool2d(1)

        # =========================================
        # ② 仿射参数预测器
        # =========================================
        # 输入：B × C × 1 × 1
        # 输出：B × (2C) × 1 × 1
        # 前 C 个通道是 scale（γ）
        # 后 C 个通道是 shift（β）
        self.affine_predictor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),      # 通道映射
            nn.ReLU(inplace=True),                                   # 非线性增强表达能力
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1)   # 生成 2C 个仿射参数
        )

    def forward(self, target_feat, reference_feat):
        # ==================================================
        # 输入说明：
        # target_feat    ：需要被调制的特征
        # reference_feat ：用于生成调制参数的参考特征
        # 两者尺寸通常为 B × C × H × W
        # ==================================================

        # ==================================================
        # Step 1：从 reference_feat 提取全局通道描述
        # ==================================================
        # 通过全局平均池化得到每个通道的全局语义统计量
        # 输出尺寸：B × C × 1 × 1
        channel_descriptor = self.global_descriptor(reference_feat)

        # ==================================================
        # Step 2：预测仿射参数
        # ==================================================
        # 生成 2C 个通道参数
        # 输出尺寸：B × 2C × 1 × 1
        affine_params = self.affine_predictor(channel_descriptor)

        # ==================================================
        # Step 3：拆分为 scale 和 shift
        # ==================================================
        # scale  ：通道缩放因子 γ
        # shift  ：通道偏移因子 β
        # 两者尺寸：B × C × 1 × 1
        scale, shift = torch.chunk(affine_params, 2, dim=1)

        # ==================================================
        # Step 4：残差式仿射调制
        # ==================================================
        # 公式：
        # y = x * (1 + γ) + β
        # 为什么用 (1 + γ)？因为这样初始状态接近恒等映射，更稳定
        calibrated_feat = target_feat * (1 + scale) + shift

        return calibrated_feat

if __name__ == "__main__":
    x1 = torch.randn(1, 32, 50, 50)
    x2 = torch.randn(1, 32, 50, 50)
    model = Geometric_Affine_Modulation(32)
    output = model(x1, x2)
    print(f"输入张量X形状: {x1.shape}")
    print(f"输入张量X形状: {x2.shape}")
    print(f"输出张量形状: {output.shape}")
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码无误~~~~")