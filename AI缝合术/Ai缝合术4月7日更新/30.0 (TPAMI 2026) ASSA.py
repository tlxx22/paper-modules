import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Function
from collections import namedtuple
from string import Template
import cupy     # idynamic implement is based on cupy-cuda
from torch.nn.modules.utils import _pair

Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


# @cupy._util.memoize(for_each_device=True)
# def load_kernel(kernel_name, code, **kwargs):
#     code = Template(code).substitute(**kwargs)
#     kernel_code = cupy.cuda.compile_with_cache(code)
#     return kernel_code.get_function(kernel_name)

@cupy._util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    return cupy.RawKernel(code, kernel_name)


CUDA_NUM_THREADS = 1024
# if you use in 3090 and above, please set 1024 for the fastest calculation
# CUDA_NUM_THREADS = 1024   # FIXME: cuda


kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS

_idynamic_kernel = kernel_loop + '''
extern "C"
__global__ void idynamic_forward_kernel(
const ${Dtype}* bottom_data, const ${Dtype}* weight_data, ${Dtype}* top_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${top_height} / ${top_width};
    const int c = (index / ${top_height} / ${top_width}) % ${channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int g = c / (${channels} / ${groups});
    ${Dtype} value = 0;
    #pragma unroll
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
        const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
        if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
          const int offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
            * ${bottom_width} + w_in;
          const int offset_weight = ((((n * ${groups} + g) * ${kernel_h} + kh) * ${kernel_w} + kw) * ${top_height} + h)
            * ${top_width} + w;
          value += weight_data[offset_weight] * bottom_data[offset];
        }
      }
    }
    top_data[index] = value;
  }
}
'''

_idynamic_kernel_backward_grad_input = kernel_loop + '''
extern "C"
__global__ void idynamic_backward_grad_input_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const weight_data, ${Dtype}* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${bottom_height} / ${bottom_width};
    const int c = (index / ${bottom_height} / ${bottom_width}) % ${channels};
    const int h = (index / ${bottom_width}) % ${bottom_height};
    const int w = index % ${bottom_width};
    const int g = c / (${channels} / ${groups});
    ${Dtype} value = 0;
    #pragma unroll
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_out_s = h + ${pad_h} - kh * ${dilation_h};
        const int w_out_s = w + ${pad_w} - kw * ${dilation_w};
        if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
          const int h_out = h_out_s / ${stride_h};
          const int w_out = w_out_s / ${stride_w};
          if ((h_out >= 0) && (h_out < ${top_height})
                && (w_out >= 0) && (w_out < ${top_width})) {
            const int offset = ((n * ${channels} + c) * ${top_height} + h_out)
                  * ${top_width} + w_out;
            const int offset_weight = ((((n * ${groups} + g) * ${kernel_h} + kh) * ${kernel_w} + kw) * ${top_height} + h_out)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
                  * ${top_width} + w_out;
            value += weight_data[offset_weight] * top_diff[offset];
          }
        }
      }
    }
    bottom_diff[index] = value;
  }
}
'''

_idynamic_kernel_backward_grad_weight = kernel_loop + '''
extern "C"
__global__ void idynamic_backward_grad_weight_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const bottom_data, ${Dtype}* const buffer_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int kh = (index / ${kernel_w} / ${top_height} / ${top_width})
          % ${kernel_h};
    const int kw = (index / ${top_height} / ${top_width}) % ${kernel_w};
    const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
    const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
    if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
      const int g = (index / ${kernel_h} / ${kernel_w} / ${top_height} / ${top_width}) % ${groups};
      const int n = (index / ${groups} / ${kernel_h} / ${kernel_w} / ${top_height} / ${top_width}) % ${num};
      ${Dtype} value = 0;
      #pragma unroll
      for (int c = g * (${channels} / ${groups}); c < (g + 1) * (${channels} / ${groups}); ++c) {
        const int top_offset = ((n * ${channels} + c) * ${top_height} + h)
              * ${top_width} + w;
        const int bottom_offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
              * ${bottom_width} + w_in;
        value += top_diff[top_offset] * bottom_data[bottom_offset];
      }
      buffer_data[index] = value;
    } else {
      buffer_data[index] = 0;
    }
  }
}
'''
class _idynamic(Function):
    @staticmethod
    def forward(ctx, input, weight, stride, padding, dilation):
        assert input.dim() == 4 and input.is_cuda
        assert weight.dim() == 6 and weight.is_cuda
        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]
        output_h = int((height + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) / stride[0] + 1)
        output_w = int((width + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) / stride[1] + 1)

        output = input.new(batch_size, channels, output_h, output_w)
        n = output.numel()

        with torch.cuda.device_of(input):
            f = load_kernel('idynamic_forward_kernel', _idynamic_kernel, Dtype=Dtype(input), nthreads=n,                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
                            num=batch_size, channels=channels, groups=weight.size()[1],
                            bottom_height=height, bottom_width=width,
                            top_height=output_h, top_width=output_w,
                            kernel_h=kernel_h, kernel_w=kernel_w,
                            stride_h=stride[0], stride_w=stride[1],
                            dilation_h=dilation[0], dilation_w=dilation[1],
                            pad_h=padding[0], pad_w=padding[1])
            f(block=(CUDA_NUM_THREADS, 1, 1),
              grid=(GET_BLOCKS(n), 1, 1),
              args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        ctx.save_for_backward(input, weight)
        ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation
        return output

    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output.contiguous()
        input, weight = ctx.saved_tensors
        stride, padding, dilation = ctx.stride, ctx.padding, ctx.dilation

        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]
        output_h, output_w = grad_output.size()[2:]

        grad_input, grad_weight = None, None

        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, channels=channels, groups=weight.size()[1],                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
                   bottom_height=height, bottom_width=width,
                   top_height=output_h, top_width=output_w,
                   kernel_h=kernel_h, kernel_w=kernel_w,
                   stride_h=stride[0], stride_w=stride[1],
                   dilation_h=dilation[0], dilation_w=dilation[1],
                   pad_h=padding[0], pad_w=padding[1])

        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(input.size())

                n = grad_input.numel()
                opt['nthreads'] = n

                f = load_kernel('idynamic_backward_grad_input_kernel',
                                _idynamic_kernel_backward_grad_input, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

            if ctx.needs_input_grad[1]:
                grad_weight = weight.new(weight.size())

                n = grad_weight.numel()
                opt['nthreads'] = n

                f = load_kernel('idynamic_backward_grad_weight_kernel',
                                _idynamic_kernel_backward_grad_weight, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return grad_input, grad_weight, None, None, None
    
def _idynamic_cuda(input, weight, bias=None, stride=1, padding=0, dilation=1):
    """ idynamic kernel
    """
    assert input.size(0) == weight.size(0)
    assert input.size(-2) // stride == weight.size(-2)
    assert input.size(-1) // stride == weight.size(-1)
    if input.is_cuda:
        out = _idynamic.apply(input, weight, _pair(stride), _pair(padding), _pair(dilation))
        if bias is not None:
            out += bias.view(1, -1, 1, 1)
    else:
        raise NotImplementedError
    return out

class IDynamicDWConv(nn.Module):

    def __init__(self, dim, kernel_size, bias):

        super(IDynamicDWConv, self).__init__()

        self.groups = dim

        self.kernel_size = kernel_size

        self.weight = nn.Conv2d(dim, dim * kernel_size ** 2, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!


    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.weight(x)
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, h, w)
        out = _idynamic_cuda(x, weight, stride=1, padding=(self.kernel_size - 1) // 2)
        return out
    
# Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

# FFN
class FeedForward(nn.Module):
    """
        GDFN in Restormer: [github] https://github.com/swz30/Restormer
    """
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()

        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
       
# Sparse Self-Attention
class SparseSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, tlc_flag=True, tlc_kernel=48, activation='relu'):                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
        super(SparseSelfAttention, self).__init__()
        self.tlc_flag = tlc_flag    # TLC flag for validation and test
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.project_in = nn.Conv2d(dim, dim * 2, 1, bias=False)
        self.dynamic_conv = IDynamicDWConv(dim * 2, kernel_size=3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.act = nn.Identity()
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'softmax':
            self.act = nn.Softmax(dim=-1)

        # [x2, x3, x4] -> [96, 72, 48]
        self.kernel_size = [tlc_kernel, tlc_kernel]

    def _forward(self, qv):
        q, v = qv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(v, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = self.act(attn)

        out = (attn @ v)

        return out

    def forward(self, x):
        b, c, h, w = x.shape

        qv = self.dynamic_conv(self.project_in(x))

        if self.training or not self.tlc_flag:
            out = self._forward(qv)
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

            out = self.project_out(out)
            return out

        # Then we use the TLC methods in test mode
        qv = self.grids(qv)  # convert to local windows
        out = self._forward(qv)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=qv.shape[-2], w=qv.shape[-1])                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
        out = self.grids_inverse(out)  # reverse

        out = self.project_out(out)
        return out

    # Code from [megvii-research/TLC] https://github.com/megvii-research/TLC
    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c // 2, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt
# ---------------------------------------------------------------------------------------------------------------------

class AttBlock(nn.Module):
    def __init__(self, dim, num_heads=6, ffn_expansion_factor=2., tlc_flag=True, tlc_kernel=48, activation='relu'):
        super(AttBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.norm2 = LayerNorm(dim, LayerNorm_type='WithBias')

        self.attn = SparseSelfAttention(dim, num_heads=num_heads, tlc_flag=tlc_flag, tlc_kernel=tlc_kernel, activation=activation, bias=False)                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
        self.ffn = FeedForward(dim, ffn_expansion_factor=ffn_expansion_factor, bias=False)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x
    


# 使用示例
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_tensor = torch.randn(1, 32, 256, 256).to(device)
    model = AttBlock(dim=32, num_heads=8).to(device)
    print(model)
    output_tensor = model(input_tensor)

    # 打印维度验证
    print("input_tensor_shape  :", input_tensor.shape)   
    print("output_tensor_shape :", output_tensor.shape)                                                                                                                            # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!
    print("\n哔哩哔哩/微信公众号: AI缝合术, 独家整理! \n")                                                                                                                           # 哔哩哔哩/微信公众号: AI缝合术, AIFengheshu 独家整理!