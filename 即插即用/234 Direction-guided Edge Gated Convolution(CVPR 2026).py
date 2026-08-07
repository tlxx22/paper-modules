import torch  # 导入 PyTorch 主库，用于张量计算
import torch.nn as nn  # 导入神经网络模块，用于构建卷积、归一化等网络层
import torch.nn.functional as F  # 导入函数式接口，用于调用 conv2d 等操作
import einops  # 导入 einops，用于方便地重排张量维度

"""
    论文地址：https://arxiv.org/abs/2603.01361
    论文题目：MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention（CVPR 2026）
    中文题目：统计特征引导的快速高分辨透射电子显微成像去噪方法（CVPR 2026）
    讲解视频：https://www.bilibili.com/video/BV1BzLn6cE2g/
    方向引导边缘门控卷积（Direction-guided Edge Gated Convolution，DEGConv）
        实际意义：①普通卷积 “全局统一处理”，无法适配局部噪声差异问题：传统卷积对图像所有位置用相同卷积操作，图像不同区域的信号波动、噪声强度差异极大，统一处理会模糊边缘并保留噪声。②不同空间位置的噪声强度和信号波动不一致问题：由于短曝光导致信噪比极低，图像中不同区域的波动情况并不相同。有些位置可能是真实结构，有些位置可能主要是随机噪声，还有一些位置是信号与噪声混合区域。
        实现方式：通过局部标准差感知每个位置的波动特征，并生成像素级动态权重来调节卷积响应，从而自适应增强结构特征、抑制局部噪声。
"""

class DirectionalStripEdgeConv(nn.Module):  # 定义方向条带边缘卷积模块
    def __init__(  # 定义模块初始化函数
        self,  # 表示当前模块对象
        in_channels,  # 输入特征图的通道数
        hidden_channels,  # 中间隐藏通道数，通常用于降低计算量
        out_channels,  # 输出特征图的通道数
        kernel_size=3,  # 条带卷积的卷积核大小，默认是 3
        bias=True  # 卷积层是否使用偏置项
    ):
        super().__init__()  # 调用父类 nn.Module 的初始化函数

        self.channel_reduction = nn.Conv2d(  # 定义 1×1 卷积，用于调整或压缩通道数
            in_channels=in_channels,  # 输入通道数
            out_channels=hidden_channels,  # 输出通道数，变为隐藏通道数
            kernel_size=1,  # 使用 1×1 卷积，只改变通道，不改变空间尺寸
            bias=bias  # 是否使用偏置项
        )

        self.horizontal_strip_conv = nn.Conv2d(  # 定义水平方向条带卷积
            in_channels=hidden_channels,  # 输入通道数
            out_channels=hidden_channels,  # 输出通道数保持不变
            kernel_size=(1, kernel_size),  # 使用 1×k 卷积核，主要提取水平方向信息
            stride=1,  # 步长为 1，保持正常滑动
            padding=(0, kernel_size // 2),  # 在宽度方向补零，保证输出尺寸不变
            groups=hidden_channels  # 使用深度卷积，每个通道独立卷积
        )

        self.vertical_strip_conv = nn.Conv2d(  # 定义垂直方向条带卷积
            in_channels=hidden_channels,  # 输入通道数
            out_channels=hidden_channels,  # 输出通道数保持不变
            kernel_size=(kernel_size, 1),  # 使用 k×1 卷积核，主要提取垂直方向信息
            stride=1,  # 步长为 1
            padding=(kernel_size // 2, 0),  # 在高度方向补零，保证输出尺寸不变
            groups=hidden_channels  # 使用深度卷积，降低参数量和计算量
        )

        self.channel_fusion = nn.Conv2d(  # 定义 1×1 卷积，用于融合水平和垂直方向特征
            in_channels=hidden_channels * 2,  # 拼接后通道数变为原来的 2 倍
            out_channels=out_channels,  # 输出通道数
            kernel_size=1,  # 使用 1×1 卷积融合通道信息
            bias=True  # 使用偏置项
        )

    def forward(self, feature_map):  # 定义前向传播函数
        reduced_feature = self.channel_reduction(feature_map)  # 先用 1×1 卷积降低或调整通道数

        horizontal_edge_feature = self.horizontal_strip_conv(reduced_feature)  # 提取水平方向边缘特征
        vertical_edge_feature = self.vertical_strip_conv(reduced_feature)  # 提取垂直方向边缘特征

        directional_edge_feature = torch.cat(  # 将两个方向的边缘特征进行拼接
            [horizontal_edge_feature, vertical_edge_feature],  # 待拼接的水平特征和垂直特征
            dim=1  # 在通道维度上拼接
        )

        output_feature = self.channel_fusion(directional_edge_feature)  # 使用 1×1 卷积融合方向边缘特征

        return output_feature  # 返回融合后的边缘特征


class HoGDirectionGuidedEdgeGateConv(nn.Module):  # 定义 HoG 方向先验引导的边缘门控卷积模块
    def __init__(  # 定义模块初始化函数
        self,  # 表示当前模块对象
        channels,  # 输入和输出特征图的通道数
        num_orientation_bins,  # HoG 方向直方图的方向划分数量
        cell_size=(8, 8)  # 每个 HoG cell 的空间大小，默认是 8×8
    ):
        super().__init__()  # 调用父类 nn.Module 的初始化函数

        self.num_orientation_bins = num_orientation_bins  # 保存方向 bin 数量
        self.cell_size = cell_size  # 保存 cell 的空间尺寸

        self.direction_embedding_encoder = nn.Sequential(  # 定义方向先验嵌入编码器
            nn.Conv2d(num_orientation_bins, channels, kernel_size=1),  # 将 HoG 方向特征映射到与输入相同的通道数
            nn.Conv2d(  # 定义深度卷积，用于进一步提取方向嵌入特征
                channels,  # 输入通道数
                channels,  # 输出通道数
                kernel_size=3,  # 使用 3×3 卷积核
                padding=1,  # padding 为 1，使空间尺寸保持不变
                groups=channels,  # 使用深度卷积，每个通道单独处理
                bias=False  # 不使用偏置项
            ),
            nn.GroupNorm(channels // 8, channels),  # 使用组归一化，稳定训练过程
            nn.ReLU(inplace=True),  # 使用 ReLU 激活函数，引入非线性
            nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化，将方向特征压缩为 1×1 的方向嵌入
        )

        self.edge_gate_generator = nn.Sequential(  # 定义门控权重生成分支
            DirectionalStripEdgeConv(  # 使用方向条带边缘卷积生成边缘响应
                in_channels=channels,  # 输入通道数
                hidden_channels=channels // 2,  # 隐藏通道数减半，降低计算量
                out_channels=channels  # 输出通道数与输入保持一致
            ),
            nn.GroupNorm(channels // 8, channels)  # 对生成的门控特征进行组归一化
        )

        self.feature_projection = nn.Sequential(  # 定义特征映射分支
            nn.Conv2d(  # 使用 1×1 卷积处理输入特征
                in_channels=channels,  # 输入通道数
                out_channels=channels,  # 输出通道数保持不变
                kernel_size=1,  # 1×1 卷积用于通道变换
                stride=1  # 步长为 1
            ),
            nn.GroupNorm(channels // 8, channels)  # 对映射后的特征进行组归一化
        )

        self.post_edge_fusion = nn.Sequential(  # 定义空间块还原后的后处理融合模块
            DirectionalStripEdgeConv(  # 再次使用方向条带边缘卷积增强边缘结构
                in_channels=channels,  # 输入通道数
                hidden_channels=channels // 2,  # 隐藏通道数减半
                out_channels=channels,  # 输出通道数保持不变
                kernel_size=3  # 条带卷积核大小为 3
            ),
            nn.GroupNorm(channels // 8, channels)  # 对输出特征进行组归一化
        )

        self.gate_activation = nn.Sigmoid()  # 定义 Sigmoid 函数，用于把门控权重限制到 0 到 1 之间

    def forward(self, input_feature):  # 定义前向传播函数
        residual_feature = input_feature  # 保存原始输入特征，用于后续残差连接

        local_patch_features = split_image_to_local_patches(input_feature)  # 将完整特征图划分成多个局部块

        hog_direction_prior = self.compute_hog_direction_prior(local_patch_features)  # 计算每个局部块的 HoG 方向先验特征
        direction_embedding = self.direction_embedding_encoder(hog_direction_prior)  # 将 HoG 方向先验编码成方向嵌入向量

        gate_weight = self.gate_activation(  # 使用 Sigmoid 得到最终门控权重
            self.edge_gate_generator(local_patch_features + direction_embedding)  # 将方向嵌入加入局部特征后生成门控响应
        )

        projected_feature = self.feature_projection(local_patch_features)  # 对局部特征进行 1×1 卷积映射
        gated_edge_feature = gate_weight * projected_feature  # 用门控权重调制局部特征，突出重要边缘区域

        restored_feature = merge_local_patches_to_image(gated_edge_feature)  # 将局部块重新拼回完整特征图

        restored_feature = restored_feature + residual_feature  # 加上原始输入特征，形成残差连接

        output_feature = self.post_edge_fusion(restored_feature)  # 对残差融合后的特征再进行边缘增强

        return output_feature  # 返回最终输出特征

    def compute_hog_direction_prior(self, patch_feature):  # 定义 HoG 方向先验计算函数
        grayscale_feature = patch_feature.mean(dim=1, keepdim=True)  # 对通道维度求平均，得到单通道灰度特征

        batch_size, _, height, width = grayscale_feature.shape  # 获取 batch 大小、高度和宽度
        device = grayscale_feature.device  # 获取当前特征所在设备，例如 CPU 或 GPU

        sobel_kernel_x = torch.tensor(  # 定义 Sobel 水平方向梯度卷积核
            [[-1, 0, 1],  # 第一行
             [-2, 0, 2],  # 第二行
             [-1, 0, 1]],  # 第三行
            dtype=torch.float32  # 设置数据类型为 float32
        ).view(1, 1, 3, 3).to(device)  # 调整为卷积核形状，并移动到输入特征所在设备

        sobel_kernel_y = torch.tensor(  # 定义 Sobel 垂直方向梯度卷积核
            [[-1, -2, -1],  # 第一行
             [0, 0, 0],  # 第二行
             [1, 2, 1]],  # 第三行
            dtype=torch.float32  # 设置数据类型为 float32
        ).view(1, 1, 3, 3).to(device)  # 调整为卷积核形状，并移动到输入特征所在设备

        gradient_x = F.conv2d(  # 计算水平方向梯度
            grayscale_feature.float(),  # 将灰度特征转为 float 类型
            sobel_kernel_x,  # 使用 Sobel x 卷积核
            padding=1  # padding 为 1，使输出尺寸不变
        )

        gradient_y = F.conv2d(  # 计算垂直方向梯度
            grayscale_feature.float(),  # 将灰度特征转为 float 类型
            sobel_kernel_y,  # 使用 Sobel y 卷积核
            padding=1  # padding 为 1，使输出尺寸不变
        )

        gradient_angle = torch.atan2(gradient_y, gradient_x)  # 根据 x 和 y 方向梯度计算每个像素的梯度角度
        gradient_angle = torch.abs(gradient_angle)  # 取绝对值，将方向角限制到非负范围

        cell_height, cell_width = self.cell_size  # 获取每个 cell 的高度和宽度

        num_cells_h = int(height / cell_height)  # 计算高度方向可以划分多少个 cell
        num_cells_w = int(width / cell_width)  # 计算宽度方向可以划分多少个 cell

        cropped_angle = gradient_angle[  # 裁剪角度图，保证高度和宽度能被 cell 尺寸整除
            :,  # 保留所有 batch
            :,  # 保留通道维度
            :num_cells_h * cell_height,  # 裁剪高度
            :num_cells_w * cell_width  # 裁剪宽度
        ]

        cell_angle_values = cropped_angle.view(  # 将角度图重排为 cell 形式
            batch_size,  # batch 大小
            num_cells_h,  # 高度方向 cell 数量
            num_cells_w,  # 宽度方向 cell 数量
            -1  # 每个 cell 内的所有像素角度展平成一维
        )

        bin_width = torch.pi / self.num_orientation_bins  # 计算每个方向 bin 对应的角度宽度

        orientation_bin_indices = (cell_angle_values / bin_width).floor().long()  # 计算每个像素角度属于哪个方向 bin
        orientation_bin_indices = torch.clamp(  # 限制 bin 索引范围，防止越界
            orientation_bin_indices,  # 原始 bin 索引
            0,  # 最小索引为 0
            self.num_orientation_bins - 1  # 最大索引为 bin 数量减 1
        )

        flattened_bin_indices = orientation_bin_indices.view(  # 将所有 cell 展平，方便逐个统计方向直方图
            batch_size * num_cells_h * num_cells_w,  # cell 总数量
            cell_angle_values.shape[-1]  # 每个 cell 内的像素数量
        )

        orientation_histograms = []  # 创建列表，用于保存每个 cell 的方向直方图

        for cell_index in range(flattened_bin_indices.shape[0]):  # 遍历每一个 cell
            current_cell_bins = flattened_bin_indices[cell_index]  # 取出当前 cell 内所有像素的方向 bin 索引

            current_histogram = torch.bincount(  # 统计当前 cell 中每个方向 bin 出现的次数
                current_cell_bins,  # 当前 cell 的 bin 索引
                minlength=self.num_orientation_bins  # 保证直方图长度等于方向 bin 数量
            )

            orientation_histograms.append(current_histogram)  # 将当前 cell 的方向直方图加入列表

        orientation_histograms = torch.stack(  # 将所有 cell 的方向直方图堆叠成张量
            orientation_histograms,  # 方向直方图列表
            dim=0  # 沿第 0 维堆叠
        ).view(  # 将展平后的 cell 维度还原为空间 cell 网格
            batch_size,  # batch 大小
            num_cells_h,  # 高度方向 cell 数量
            num_cells_w,  # 宽度方向 cell 数量
            self.num_orientation_bins  # 每个 cell 的方向 bin 数量
        )

        orientation_histograms = orientation_histograms / (cell_height * cell_width)  # 对直方图计数做归一化

        bin_center_start = torch.pi / (2 * self.num_orientation_bins)  # 计算第一个方向 bin 的中心角度

        orientation_bin_centers = torch.linspace(  # 生成所有方向 bin 的中心角度
            bin_center_start,  # 起始中心角度
            torch.pi - bin_center_start,  # 结束中心角度
            self.num_orientation_bins  # 生成的中心角数量
        ).to(device)  # 将中心角张量移动到输入特征所在设备

        orientation_bin_centers = orientation_bin_centers.repeat(  # 扩展方向中心角，使其与直方图形状一致
            batch_size,  # batch 大小
            num_cells_h,  # 高度方向 cell 数量
            num_cells_w,  # 宽度方向 cell 数量
            1  # 方向 bin 维度保持不变
        )

        hog_direction_prior = orientation_bin_centers * orientation_histograms  # 用方向中心角加权方向直方图，得到 HoG 方向先验

        hog_direction_prior = hog_direction_prior.permute(0, 3, 1, 2)  # 调整维度为 [B, num_bins, H_cells, W_cells]

        return hog_direction_prior  # 返回 HoG 方向先验特征


def split_image_to_local_patches(feature_map):  # 定义函数：将完整特征图划分为局部块
    patch_feature = einops.rearrange(  # 使用 einops 对张量维度进行重排
        feature_map,  # 输入特征图，形状为 [B, C, H, W]
        'b c (num_h h) (num_w w) -> (num_h num_w b) c h w',  # 将 H 和 W 分成 2×2 个局部块，并把块数合并到 batch 维度
        num_h=2,  # 高度方向划分成 2 块
        num_w=2  # 宽度方向划分成 2 块
    )

    return patch_feature  # 返回局部块特征，形状为 [4B, C, H/2, W/2]


def merge_local_patches_to_image(patch_feature):  # 定义函数：将局部块重新拼回完整图像
    feature_map = einops.rearrange(  # 使用 einops 对张量维度进行反向重排
        patch_feature,  # 输入局部块特征，形状为 [4B, C, H/2, W/2]
        '(num_h num_w b) c h w -> b c (num_h h) (num_w w)',  # 将局部块重新拼接回完整空间尺寸
        num_h=2,  # 高度方向有 2 个局部块
        num_w=2  # 宽度方向有 2 个局部块
    )

    return feature_map  # 返回完整特征图，形状为 [B, C, H, W]


if __name__ == "__main__":  # 判断当前文件是否作为主程序运行
    input_tensor = torch.randn(1, 32, 64, 64)  # 随机生成一个输入特征图，形状为 [1, 32, 64, 64]
    model = HoGDirectionGuidedEdgeGateConv(  # 实例化 HoG 方向先验引导的边缘门控卷积模块
        channels=32,  # 输入和输出通道数设置为 32
        num_orientation_bins=8  # 将方向划分为 8 个 bin
    )
    output_tensor = model(input_tensor)  # 将输入特征送入模型，得到输出特征
    print("input_tensor_shape  :", input_tensor.shape)  # 打印输入特征图的形状
    print("output_tensor_shape :", output_tensor.shape)  # 打印输出特征图的形状
    print("微信公众号、B站、CSDN同号")
    print("布尔大学士 提醒您：代码完毕，逻辑无误~~~~")