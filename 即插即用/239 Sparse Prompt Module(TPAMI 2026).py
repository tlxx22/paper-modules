import torch
import torch.nn as nn
import torch.nn.functional as F
"""
    论文地址：https://arxiv.org/pdf/2504.09973
    论文题目：Beyond Degradation Redundancy: Contrastive Prompt Learning for All-in-One Image Restoration (TPAMI 2026)
    中文题目：突破退化冗余：面向全能图像修复的对比提示学习(TPAMI 2026)
    讲解视频：https://www.bilibili.com/video/BV12qGk6cEvo/
    稀疏提示模块（Sparse Prompt Module，SPM）
        实际意义：①不同退化任务的 Prompt 表征相互重叠：不同退化特征之间存在一定相似性，例如：去雨与去噪都涉及局部高频；去雾与亮度增强都涉及全局亮度、对比度；去雨雪都需要去除稀疏伪影。
                    因此，传统端到端自适应 Prompt 学习容易让不同 Prompt 学到相似的退化信息，模型难以形成“哪个Prompt专门负责哪类退化”的明确分工。
                ②Prompt 缺少明确的专业化分工：传统密集 Prompt 机制中，每个 Prompt 都可能参与多种退化任务的复原过程，容易导致：单个 Prompt 缺少明确职责；不同Prompt学到重复信息；任务之间的干扰更加明显。
        实现方式：通过稀疏路由动态选择最匹配的任务 Prompt，从而减少退化表征冗余并提升效果。
"""

# 定义稀疏提示生成模块，对应论文中的 Sparse Prompt Module（SPM）
class SparsePromptModule(nn.Module):

    # 初始化模块中需要使用的参数和网络层
    def __init__(
        self,
        prompt_channels=128,
        num_prompt_experts=5,
        prompt_spatial_size=96,
        router_input_dim=192,
        num_active_experts=1
    ):
        # 调用父类 nn.Module 的初始化方法，使当前类具备神经网络模块功能
        super().__init__()

        # 创建一个可以被网络训练更新的 Prompt 专家库
        # 第 1 个维度表示共享的初始 batch 维，后续会根据实际 batch 扩展
        # 第 2 个维度表示一共有多少个候选 Prompt 专家
        # 第 3 个维度表示每个 Prompt 专家的通道数量
        # 第 4、5 个维度表示每个 Prompt 专家的空间尺寸
        self.prompt_expert_bank = nn.Parameter(
            # 使用随机数初始化所有可学习的 Prompt 专家模板
            torch.rand(
                # 初始 batch 维度设置为 1，表示所有样本共享这一套可学习模板
                1,
                # Prompt 专家的总数量
                num_prompt_experts,
                # 每个 Prompt 专家的通道数量
                prompt_channels,
                # 每个 Prompt 专家的高度
                prompt_spatial_size,
                # 每个 Prompt 专家的宽度
                prompt_spatial_size
            ),
            # 设置为 True，表示该 Prompt 专家库会在训练过程中更新
            requires_grad=True
        )

        # 定义路由器 Router，本质上是一个全连接层
        # 它会根据输入图像特征判断当前样本更适合使用哪些 Prompt 专家
        self.prompt_router = nn.Linear(
            # 输入维度对应全局特征向量的通道数
            router_input_dim,
            # 输出维度对应所有 Prompt 专家的数量
            num_prompt_experts
        )

        # 定义一个 3×3 卷积层，用于进一步处理组合得到的 Prompt 特征
        self.prompt_refinement_conv = nn.Conv2d(
            # 输入 Prompt 的通道数
            prompt_channels,
            # 输出 Prompt 的通道数保持不变
            prompt_channels,
            # 使用 3×3 卷积核提取局部空间信息
            kernel_size=3,
            # 步长为 1，保持空间信息连续传递
            stride=1,
            # 填充为 1，使卷积前后的空间尺寸不变
            padding=1,
            # 不额外引入偏置项
            bias=False
        )

        # 保存每个输入样本需要激活的 Prompt 专家数量
        # 论文中默认使用 top-k 稀疏选择，k 越小，提示选择越稀疏
        self.num_active_experts = num_active_experts

        # 保存 Prompt 专家库中一共有多少个候选专家
        self.num_prompt_experts = num_prompt_experts

    # 定义前向传播过程，即输入特征如何生成对应的 Prompt
    def forward(self, input_features, generate_negative_prompt=False):

        batch_size, channels, height, width = input_features.shape

        # 对输入特征图的高度维和宽度维求平均
        # 该操作相当于全局平均池化，用一个向量概括整张特征图的整体信息
        # 输入形状为 [B, C, H, W]，输出形状变为 [B, C]
        global_feature_descriptor = input_features.mean(dim=(-2, -1))

        # 将每个样本的全局特征向量输入 Router
        # Router 为每一个 Prompt 专家预测一个匹配分数
        router_logits = self.prompt_router(global_feature_descriptor)
        # 对所有 Prompt 专家的分数执行 softmax
        # softmax 会把分数转换为概率形式，并保证每个样本的概率总和为 1
        routing_probabilities = F.softmax(
            # 输入 Router 产生的专家匹配分数
            router_logits,
            # 在 Prompt 专家这一维度上计算概率分布
            dim=1
        ).to(input_features.dtype)

        # 从所有 Prompt 专家概率中选出数值最大的 top-k 个专家
        # 第一个返回值是被选中专家的概率值，这里不单独保存
        # 第二个返回值是被选中专家的位置编号
        _, active_expert_indices = torch.topk(
            # 输入所有 Prompt 专家的路由概率
            routing_probabilities,
            # 指定需要选择几个最匹配的专家
            self.num_active_experts,
            # 沿着专家数量这一维进行 top-k 选择
            dim=1
        )

        # 判断当前是否需要生成负 Prompt
        # False 表示生成正常匹配当前输入的正 Prompt
        if not generate_negative_prompt:
            # 创建一个与 routing_probabilities 形状完全相同的全零张量
            # 它将用于保存稀疏化后的专家权重
            sparse_routing_weights = torch.zeros_like(routing_probabilities)
            # 根据 active_expert_indices 找到被选中的专家对应概率
            active_expert_weights = routing_probabilities.gather(
                # 沿着专家维度读取权重
                1,
                # 提供 top-k 专家的位置编号
                active_expert_indices
            )
            # 将 top-k 专家的概率填写到稀疏权重矩阵对应位置
            # 没有被选中的专家仍然保持为 0
            sparse_routing_weights.scatter_(
                # 在专家维度上填写数值
                dim=1,
                # 指定需要填写的位置，即被激活的专家编号
                index=active_expert_indices,
                # 指定填写的内容，即被激活专家原本的路由概率
                src=active_expert_weights
            )
            # 使用保留下来的稀疏权重，对 Prompt 专家库进行加权组合
            # 得到与当前输入最匹配的正 Prompt
            generated_prompt = self._compose_prompt_from_experts(
                # 输入当前样本对应的稀疏专家权重
                sparse_routing_weights,
                # 输入当前批次包含的样本数量
                batch_size
            )

        # True 表示需要生成与当前输入不匹配的负 Prompt
        else:
            # 在负 Prompt 采样过程中暂时关闭梯度记录
            # 这样负专家选择过程不会参与梯度更新
            with torch.no_grad():
                # 创建一个布尔类型的候选专家掩码
                # 初始情况下，所有 Prompt 专家都可以作为负候选专家
                negative_candidate_mask = torch.ones_like(
                    # 掩码张量的形状与路由概率张量保持一致
                    routing_probabilities,
                    # 将张量类型设置为布尔类型，只保存 True 或 False
                    dtype=torch.bool
                )
                # 将正分支中已经选中的 top-k 专家位置设置为 False
                # 这样可以避免把与当前输入最匹配的专家当作负 Prompt
                negative_candidate_mask.scatter_(
                    # 在专家维度上修改掩码
                    dim=1,
                    # 找到正分支中被激活的专家位置
                    index=active_expert_indices,
                    # 将这些位置设置为不可用于负采样
                    value=False
                )
                # 为每个 Prompt 专家随机生成一个采样分数
                # 后续会利用随机分数，从剩余专家中随机选择负专家
                random_sampling_scores = torch.rand_like(routing_probabilities)
                # 将不允许作为负 Prompt 的专家位置赋值为负无穷
                # 这样这些专家在后续 top-k 操作中不会被选中
                random_sampling_scores.masked_fill_(
                    # 取反后，True 表示正专家所在的位置
                    ~negative_candidate_mask,
                    # 将正专家位置的随机分数设置为负无穷
                    -float("inf")
                )
                # 从允许选择的剩余专家中，随机选出指定数量的负专家
                # 第一个返回值为随机分数，这里不需要单独保存
                # 第二个返回值为负专家的位置编号
                _, negative_expert_indices = torch.topk(
                    # 输入经过掩码处理后的随机采样分数
                    random_sampling_scores,
                    # 负 Prompt 使用的专家数量与正 Prompt 保持一致
                    self.num_active_experts,
                    # 沿着专家维度进行选择
                    dim=1
                )
                # 根据负专家的位置编号，从原始路由概率中取出对应权重
                # 这些权重用于组合负 Prompt 专家模板
                negative_expert_weights = routing_probabilities.gather(
                    # 沿着专家维度读取
                    1,
                    # 读取被随机选择出的负专家位置
                    negative_expert_indices
                )
                # 创建一个全零张量，用于保存负 Prompt 对应的稀疏专家权重
                sparse_routing_weights = torch.zeros_like(routing_probabilities)
                # 只在负专家对应的位置写入权重
                # 其余没有被选择的专家仍然保持为 0
                sparse_routing_weights.scatter_(
                    # 在专家维度上填写负权重
                    dim=1,
                    # 指定负专家所在的位置
                    index=negative_expert_indices,
                    # 写入这些负专家原本对应的路由概率
                    src=negative_expert_weights
                )
                # 使用随机选中的负专家组合生成负 Prompt
                generated_prompt = self._compose_prompt_from_experts(
                    # 输入负 Prompt 对应的稀疏路由权重
                    sparse_routing_weights,
                    # 输入当前批次大小
                    batch_size
                ).detach()

        # 将生成的 Prompt 空间尺寸调整为与输入特征图一致
        # 这样后续网络才能方便地将 Prompt 与输入特征进行融合
        generated_prompt = F.interpolate(generated_prompt,size=(height, width),mode="bilinear",align_corners=False)
        # 将尺寸调整后的 Prompt 输入 3×3 卷积层：用于进一步提取局部信息并细化 Prompt 表达
        generated_prompt = self.prompt_refinement_conv(generated_prompt)

        # 返回最终生成的 Prompt 特征图
        return generated_prompt

    # 定义一个辅助函数，用于根据稀疏权重组合 Prompt 专家库
    def _compose_prompt_from_experts(self, sparse_routing_weights, batch_size):
        # 在稀疏权重张量末尾连续增加三个维度
        # 原始形状为 [B, N]
        # 扩展后形状为 [B, N, 1, 1, 1]
        # 这样每一个专家权重就可以作用于完整的 Prompt 特征图
        expanded_routing_weights = sparse_routing_weights.unsqueeze(-1) .unsqueeze(-1).unsqueeze(-1)

        # 将共享的 Prompt 专家库按照当前批次大小进行扩展
        # 原始形状为 [1, N, C, H, W]
        # 扩展后形状为 [B, N, C, H, W]
        # 每个样本都可以使用同一套专家库，但选择权重可以不同
        batched_prompt_experts = self.prompt_expert_bank.expand(
            # 将第一个维度扩展为当前 batch 中的样本数量
            batch_size,
            # -1 表示保持 Prompt 专家数量不变
            -1,
            # -1 表示保持通道数量不变
            -1,
            # -1 表示保持 Prompt 高度不变
            -1,
            # -1 表示保持 Prompt 宽度不变
            -1
        )

        # 将每个 Prompt 专家模板乘以当前样本分配给它的路由权重
        # 没有被选中的专家权重为 0，因此不会影响最终结果
        weighted_prompt_experts = expanded_routing_weights * batched_prompt_experts

        # 沿着 Prompt 专家数量这一维进行求和
        # 将若干个被激活的 Prompt 专家融合成一个最终 Prompt
        # 输出形状为 [B, C, H, W]
        combined_prompt = torch.sum(
            # 输入完成加权后的全部 Prompt 专家特征
            weighted_prompt_experts,
            # 在专家数量所在的维度进行求和
            dim=1
        )

        # 返回组合后的 Prompt 特征
        return combined_prompt

if __name__ == "__main__":
    # [1, 64, 32, 32] 表示 batch 中有 1 个样本、64 个通道、空间尺寸为 32×32
    input_features = torch.randn(1, 64, 32, 32)
    # 创建一个稀疏提示生成模块实例
    sparse_prompt_module = SparsePromptModule(
        # 设置每个 Prompt 专家包含 64 个通道
        prompt_channels=64,
        # 设置 Prompt 专家库中共有 5 个候选专家
        num_prompt_experts=5,
        # 设置每个初始 Prompt 专家的空间尺寸为 64×64
        prompt_spatial_size=64,
        # 设置 Router 输入的全局特征向量维度为 64
        router_input_dim=64,
        # 设置每次只激活 1 个 Prompt 专家
        num_active_experts=1
    )
    # 正 Prompt 是模型认为“最适合处理当前图像”的提示特征；【只使用这个就可以】
    positive_prompt = sparse_prompt_module(input_features,generate_negative_prompt=False)
    # 负 Prompt 是选择的“与当前图像不太匹配”的提示特征。
    # negative_prompt = sparse_prompt_module(input_features,generate_negative_prompt=True)
    print("输入特征维度：", input_features.shape)
    print("正 Prompt 维度：", positive_prompt.shape)
    # print("负 Prompt 维度：", negative_prompt.shape)
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")