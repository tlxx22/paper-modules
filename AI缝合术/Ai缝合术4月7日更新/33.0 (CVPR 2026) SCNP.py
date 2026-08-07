import torch
from torch.nn import functional as F


class SCNP(torch.nn.Module):
    def __init__(self, dimensions, neighborhood_size=3):
        super(SCNP, self).__init__()

        if dimensions == "2D":
            self.mp = torch.nn.functional.max_pool2d
            self.ns = (neighborhood_size, neighborhood_size)
            self.st = (1, 1)
            self.pad = (neighborhood_size//2, neighborhood_size//2)
        elif dimensions == "3D":
            self.mp = torch.nn.functional.max_pool3d
            self.ns = (neighborhood_size, neighborhood_size, neighborhood_size)
            self.st = (1, 1, 1)
            self.pad = (neighborhood_size//2, neighborhood_size//2, neighborhood_size//2)
        else:
            raise Exception("`dimensions` parameters must be either '2D' or '3D'")

    def forward(self, logits, target):
        assert logits.shape == target.shape, "`target` should be one-hot encoded and have the same shape as `logits`"                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!

        # MinPooling in the foreground
        t1 = -self.mp(-(logits*target+9999*(1-target)), self.ns, self.st, self.pad)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
        # MaxPooling in the background
        t2 = self.mp((logits*(1-target)-9999*target), self.ns, self.st, self.pad)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
        z_tilde = t1*target + t2*(1-target)

        return z_tilde

# 定义模型(你的实际模型)
class model(torch.nn.Module):
    def __init__(self, x):
        super(model, self).__init__()
        self.x = x

    def forward(self, x):
        return x

device = "cuda" if torch.cuda.is_available() else "cpu"
# 数据
x= torch.randn(1, 64, 32, 32).to(device)
# 标签GT
y = torch.randn(1, 64, 32, 32).to(device)

model = model(x)
z_logits = model(x)


scnp = SCNP(dimensions="2D")
# 可以对3D数据计算损失约束
# # scnp = SCNP("3D", 5)
scnp_logits = scnp(z_logits, y)

# 计算损失
loss = torch.nn.CrossEntropyLoss(F.sigmoid(scnp_logits), y) # Softmax Or Sigmoid

# loss.backward()...反向传播