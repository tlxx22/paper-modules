import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

class RectifyCoordsGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coords, coords_lambda=20):
        ctx.in1 = coords_lambda
        ctx.save_for_backward(coords)
        return coords

    @staticmethod
    def backward(ctx, grad_output):
        coords_lambda = ctx.in1
        coords, = ctx.saved_tensors
        grad_output[coords < -1.001] += -coords_lambda * 10
        grad_output[coords > 1.001] += coords_lambda * 10
        return grad_output, None
    
def calc_rel_pos_spatial(
    attn,
    q,
    q_shape,
    k_shape,
    rel_pos_h,
    rel_pos_w,
    overlap=0
    ):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    k_h = k_h + 2 * overlap
    k_w = k_w + 2 * overlap

    # Scale up rel pos if shapes for q and k are different.
    # q_h_ratio = max(k_h / q_h, 1.0)
    # k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h)[:, None] - torch.arange(k_h)[None, :]
    )
    dist_h += (k_h - 1)
    # q_w_ratio = max(k_w / q_w, 1.0)
    # k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w)[:, None] - torch.arange(k_w)[None, :]
    )
    dist_w += (k_w - 1)

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn

class QuadrangleAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., window_size=7, rpe='v2', coords_lambda=20):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.window_num = 1
        self.coords_lambda = coords_lambda

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.identity = nn.Identity()  # for hook
        self.identity_attn = nn.Identity()  # for hook
        self.identity_distance = nn.Identity()

        self.transform = nn.Sequential(
                nn.AvgPool2d(kernel_size=window_size, stride=window_size), 
                nn.LeakyReLU(),
                nn.Conv2d(dim, self.num_heads*9, kernel_size=1, stride=1)
            )

        self.rpe = rpe
        if rpe == 'v1':
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(# 微信公众号:AI缝合术
                torch.zeros((window_size * 2 - 1) * (window_size * 2 - 1), num_heads))  # (2*Wh-1 * 2*Ww-1 + 1, nH) # 微信公众号:AI缝合术
            # self.relative_position_bias = torch.zeros(1, num_heads) # the extra is for the token outside windows
            # 微信公众号:AI缝合术
            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size)
            coords_w = torch.arange(window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size - 1
            relative_coords[:, :, 0] *= 2 * window_size - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)
            # 微信公众号:AI缝合术
            trunc_normal_(self.relative_position_bias_table, std=.02)
            print('The v1 relative_pos_embedding is used')

        elif rpe == 'v2':
            q_size = window_size
            rel_sp_dim = 2 * q_size - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            trunc_normal_(self.rel_pos_h, std=.02)# 微信公众号:AI缝合术
            trunc_normal_(self.rel_pos_w, std=.02)
            print('The v2 relative_pos_embedding is used')

    def forward(self, x, h, w):
        b, N, C = x.shape
        x = x.reshape(b, h, w, C).permute(0, 3, 1, 2)
        shortcut = x
        qkv_shortcut = F.conv2d(shortcut, self.qkv.weight.unsqueeze(-1).unsqueeze(-1), bias=self.qkv.bias, stride=1)
        ws = self.window_size
        padding_t = 0
        padding_d = (ws - h % ws) % ws
        padding_l = 0
        padding_r = (ws - w % ws) % ws
        expand_h, expand_w = h+padding_t+padding_d, w+padding_l+padding_r
        window_num_h = expand_h // ws
        window_num_w = expand_w // ws
        assert expand_h % ws == 0
        assert expand_w % ws == 0
        image_reference_h = torch.linspace(-1, 1, expand_h).to(x.device)
        image_reference_w = torch.linspace(-1, 1, expand_w).to(x.device)# 微信公众号:AI缝合术
        image_reference = torch.stack(torch.meshgrid(image_reference_w, image_reference_h), 0).permute(0, 2, 1).unsqueeze(0) # 2, h, w
        window_reference = nn.functional.avg_pool2d(image_reference, kernel_size=ws)
        image_reference = image_reference.reshape(1, 2, window_num_h, ws, window_num_w, ws)
        window_center_coords = window_reference.reshape(1, 2, window_num_h, 1, window_num_w, 1)

        base_coords_h = torch.arange(ws).to(x.device) * 2 / (expand_h-1)
        base_coords_h = (base_coords_h - base_coords_h.mean())
        base_coords_w = torch.arange(ws).to(x.device) * 2 / (expand_w-1)
        base_coords_w = (base_coords_w - base_coords_w.mean())


        expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(window_num_h, 1)
        assert expanded_base_coords_h.shape[0] == window_num_h
        assert expanded_base_coords_h.shape[1] == ws
        expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(window_num_w, 1)
        assert expanded_base_coords_w.shape[0] == window_num_w
        assert expanded_base_coords_w.shape[1] == ws# 微信公众号:AI缝合术
        expanded_base_coords_h = expanded_base_coords_h.reshape(-1)
        expanded_base_coords_w = expanded_base_coords_w.reshape(-1)
        window_coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2, 1).reshape(1, 2, window_num_h, ws, window_num_w, ws).permute(0, 2, 4, 1, 3, 5)
        # base_coords = image_reference

        qkv = qkv_shortcut
        qkv = torch.nn.functional.pad(qkv, (padding_l, padding_r, padding_t, padding_d))
        qkv = rearrange(qkv, 'b (num h dim) hh ww -> num (b h) dim hh ww', h=self.num_heads//self.window_num, num=3, dim=self.dim//self.num_heads, b=b, hh=expand_h, ww=expand_w)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        if h > ws or w > ws:
            # getting the learned params for the varied windows and the coordinates of each pixel# 微信公众号:AI缝合术
            x = torch.nn.functional.pad(shortcut, (padding_l, padding_r, padding_t, padding_d))
            sampling_ = self.transform(x).reshape(b*self.num_heads//self.window_num, 9, window_num_h, window_num_w).permute(0, 2, 3, 1)
            sampling_offsets = sampling_[..., :2,]
            sampling_offsets[..., 0] = sampling_offsets[..., 0] / (expand_w // ws)
            sampling_offsets[..., 1] = sampling_offsets[..., 1] / (expand_h // ws)
            # sampling_offsets = sampling_offsets.permute(0, 3, 1, 2)
            sampling_offsets = sampling_offsets.reshape(-1, window_num_h, window_num_w, 2, 1)
            sampling_scales = sampling_[..., 2:4] + 1
            sampling_shear = sampling_[..., 4:6]
            sampling_projc = sampling_[..., 6:8]
            sampling_rotation = sampling_[..., -1]
            zero_vector = torch.zeros(b*self.num_heads//self.window_num, window_num_h, window_num_w).cuda()
            sampling_projc = torch.cat([
                sampling_projc.reshape(-1, window_num_h, window_num_w, 1, 2),
                torch.ones_like(zero_vector).cuda().reshape(-1, window_num_h, window_num_w, 1, 1)
                ], dim=-1)

            shear_matrix = torch.stack([
                torch.ones_like(zero_vector).cuda(),
                sampling_shear[..., 0],
                sampling_shear[..., 1],
                torch.ones_like(zero_vector).cuda()], dim=-1).reshape(-1, window_num_h, window_num_w, 2, 2)
            scales_matrix = torch.stack([
                sampling_scales[..., 0],
                torch.zeros_like(zero_vector).cuda(),
                torch.zeros_like(zero_vector).cuda(),
                sampling_scales[..., 1],
            ], dim=-1).reshape(-1, window_num_h, window_num_w, 2, 2)
            rotation_matrix = torch.stack([
                sampling_rotation.cos(),
                sampling_rotation.sin(),
                -sampling_rotation.sin(),
                sampling_rotation.cos()
            ], dim=-1).reshape(-1, window_num_h, window_num_w, 2, 2)
            basic_transform_matrix = rotation_matrix @ shear_matrix @ scales_matrix
            affine_matrix = torch.cat(
                (torch.cat((basic_transform_matrix, sampling_offsets), dim=-1), sampling_projc), dim=-2)
            window_coords_pers = torch.cat([
                window_coords.flatten(-2, -1), torch.ones(1, window_num_h, window_num_w, 1, ws*ws).cuda()
            ], dim=-2)
            transform_window_coords = affine_matrix @ window_coords_pers
            # transform_window_coords = rotation_matrix @ shear_matrix @ scales_matrix @ window_coords.flatten(-2, -1)
            _transform_window_coords3 = transform_window_coords[..., -1, :]
            _transform_window_coords3[_transform_window_coords3==0] = 1e-6
            transform_window_coords = transform_window_coords[..., :2, :] / _transform_window_coords3.unsqueeze(dim=-2)
            # _transform_window_coords0 = transform_window_coords[..., 0, :] / _transform_window_coords3
            # _transform_window_coords1 = transform_window_coords[..., 1, :] / _transform_window_coords3# 微信公众号:AI缝合术
            # transform_window_coords = torch.stack((_transform_window_coords0, _transform_window_coords1), dim=-2)
            # transform_window_coords = transform_window_coords[..., :2, :]# 微信公众号:AI缝合术
            transform_window_coords_distance = transform_window_coords.reshape(-1, window_num_h, window_num_w, 2, ws*ws, 1)
            transform_window_coords_distance = transform_window_coords_distance - window_coords.reshape(-1, window_num_h, window_num_w, 2, 1, ws*ws)
            transform_window_coords_distance = torch.sqrt((transform_window_coords_distance[..., 0, :, :]*(expand_w-1)/2) ** 2 + (transform_window_coords_distance[..., 1, :, :]*(expand_h-1)/2) ** 2)
            transform_window_coords_distance = rearrange(transform_window_coords_distance, '(b h) hh ww n1 n2 -> (b hh ww) h n1 n2', b=b, h=self.num_heads, hh=window_num_h, ww=window_num_w, n1=ws*ws, n2=ws*ws)
            transform_window_coords = transform_window_coords.reshape(-1, window_num_h, window_num_w, 2, ws, ws).permute(0, 3, 1, 4, 2, 5)
            #TODO: adjust the order of transformation

            coords = window_center_coords.repeat(b*self.num_heads, 1, 1, 1, 1, 1) + transform_window_coords

            # coords = base_coords.repeat(b*self.num_heads//self.window_num, 1, 1, 1, 1, 1) + window_coords * sampling_scales[:, :, :, None, :, None] + sampling_offsets[:, :, :, None, :, None]
            sample_coords = coords.permute(0, 2, 3, 4, 5, 1).reshape(b*self.num_heads, ws*window_num_h, ws*window_num_w, 2)
            sample_coords = RectifyCoordsGradient.apply(sample_coords, self.coords_lambda)
            _sample_coords = self.identity(sample_coords)

            k_selected = F.grid_sample(k, grid=sample_coords, padding_mode='zeros', align_corners=True)
            v_selected = F.grid_sample(v, grid=sample_coords, padding_mode='zeros', align_corners=True)

            q = rearrange(q, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)
            # k = k_selected.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.dim//self.num_heads)
            k = rearrange(k_selected, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)
            # v = v_selected.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.dim//self.num_heads)
            v = rearrange(v_selected, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)
        else:
            transform_window_coords_distance = None
            q = rearrange(q, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)
            # k = k_selected.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.dim//self.num_heads)
            k = rearrange(k, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)
            # v = v_selected.reshape(b, self.num_heads, self.dim//self.num_heads, window_num_h, self.attn_ws, window_num_w, self.attn_ws).permute(0, 3, 5, 1, 4, 6, 2).reshape(b*window_num_h*window_num_w, self.num_heads, self.attn_ws*self.attn_ws, self.dim//self.num_heads)
            v = rearrange(v, '(b h) dim (hh ws1) (ww ws2) -> (b hh ww) h (ws1 ws2) dim', b=b, h=self.num_heads//self.window_num, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=ws, ws2=ws)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.rpe == 'v1':
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn += relative_position_bias.unsqueeze(0)
            pass
        elif self.rpe == 'v2':
            # q = rearrange(q, '(b hh ww) h (ws1 ws2) dim -> b h (hh ws1 ww ws2) dim', b=b, h=self.num_heads, dim=self.dim//self.num_heads, ww=window_num_w, hh=window_num_h, ws1=self.window_size, ws2=self.window_size)
            # with torch.cuda.amp.autocast(enable=False):
            attn = calc_rel_pos_spatial(attn.float(), q.float(), (self.window_size, self.window_size), (self.window_size, self.window_size), self.rel_pos_h.float(), self.rel_pos_w.float())
        attn = attn.softmax(dim=-1)
        _attn = self.identity_attn(rearrange(attn, '(b hh ww) h ws1 ws2 -> (b h) (hh ww) ws1 ws2', b=b, h=self.num_heads//self.window_num, ww=window_num_w, hh=window_num_h, ws1=ws**2, ws2=ws**2))
        if transform_window_coords_distance is not None:
            transform_window_coords_distance = (transform_window_coords_distance * attn).sum(dim=-1)
            transform_window_coords_distance = self.identity_distance(transform_window_coords_distance)

        out = attn @ v
        out = rearrange(out, '(b hh ww) h (ws1 ws2) dim -> b (h dim) (hh ws1) (ww ws2)', h=self.num_heads//self.window_num, b=b, hh=window_num_h, ww=window_num_w, ws1=ws, ws2=ws)
        if padding_t + padding_d + padding_l + padding_r > 0:
            out = out[:, :, padding_t:h+padding_t, padding_l:w+padding_l]
        # globel_out.append(out)
        
        # globel_out = torch.stack(globel_out, dim=0)
        # out = rearrange(out, 'b c hh ww -> b (wsnum c) hh ww', wsnum=1, c=self.dim, b=b, hh=h, ww=w)
        out = out.reshape(b, self.dim, -1).permute(0, 2, 1)
        out = self.proj(out)
        return out

    def _reset_parameters(self):
        nn.init.constant_(self.transform[-1].weight, 0.)
        nn.init.constant_(self.transform[-1].bias, 0.)


if __name__ == "__main__":

# 定义输入张量的尺寸 (batch_size, seq_len, channels)
    batch_size = 4
    seq_len_h = 64  # height
    seq_len_w = 64  # width
    channels = 256
    num_heads = 8
    window_size = 7

    # 创建随机输入张量 x (batch_size, seq_len_h * seq_len_w, channels) 和 h, w
    x = torch.randn(batch_size, seq_len_h * seq_len_w, channels).cuda()
    
    # 定义 QuadrangleAttention 模型
    model = QuadrangleAttention(dim=channels, num_heads=num_heads, window_size=window_size, rpe='v2').cuda()
    print(model)
    print("\n微信公众号: AI缝合术!\n")
    
    # 调用模型
    out = model(x, seq_len_h, seq_len_w)
    
    # 打印输出张量的尺寸
    print(f'输入 x 的尺寸: {x.size()}')
    print(f'输出 out 的尺寸: {out.size()}')
