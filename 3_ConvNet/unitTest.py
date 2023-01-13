import torch
from torch import nn
from torch.utils.cpp_extension import load

import math

conv2d_cuda = load(name="conv2d_cuda",
                    sources=["conv2d.cpp", "conv2d_kernel.cu"],
                    extra_cflags=['-fopenmp'],
                    extra_include_paths=['../common'],
                    extra_ldflags=['-L/home/seonghoon/Workspace/IITP-PIM-SW/PIM-Application/common -lopenblas'],
                    verbose=True)

# Parameters
i_b = 128
i_w = 4
i_h = 4
i_c = 1
k_w = 3
k_h = 3
o_c = 3
padding = 1
stride = 1
o_w = math.floor(((i_w - k_w + 2*padding) / stride) + 1)
o_h = math.floor(((i_h - k_h + 2*padding) / stride) + 1)

# [SSH] Temporarily commented out for CPU execution
# train_on_gpu = torch.cuda.is_available()
train_on_gpu = False
device = "cuda" if train_on_gpu else "cpu"

x = torch.rand(i_b, i_c, i_w, i_h, generator=torch.Generator().manual_seed(123)).to(device)
conv1 = nn.Conv2d(in_channels=i_c, out_channels=o_c, kernel_size=(k_w, k_h), stride=stride, padding=padding, bias=False).to(device)
y = conv1(x)
print(f'input shape: {x.shape}')
print(f'weight shape: {conv1.weight.shape}') # cf. tensor.shape is an alias for tensor.size()
print(f'output shape: {y.shape}')

unfold = nn.Unfold(kernel_size=(k_w, k_h), padding=padding, stride=stride)
x_unfold = unfold(x)
x_unfold_transpose = x_unfold.transpose(1, 2)
weight = conv1.weight
weight_transpose = weight.view(weight.size(0), -1).t()
output_unfold = x_unfold_transpose.matmul(weight_transpose)
output_unfold_transpose = output_unfold.transpose(1, 2)
fold = nn.Fold(output_size=(o_w, o_h), kernel_size=(1, 1))
final = fold(output_unfold_transpose)
diff = (y - final).abs().max()

print(f'[VERBOSE] im2col shape: {x_unfold.shape}, {x_unfold.is_contiguous()}')
print(f'[VERBOSE] im2col transposed shape: {x_unfold_transpose.shape}, {x_unfold_transpose.is_contiguous()}')
print(f'[VERBOSE] weight shape: {weight.shape}, {weight.is_contiguous()}')
print(f'[VERBOSE] weight transposed shape: {weight_transpose.shape}, {weight_transpose.is_contiguous()}')
print(f'[VERBOSE] output unfold shape: {output_unfold.shape}, {output_unfold.is_contiguous()}')
print(f'[VERBOSE] output_unfold_transpose shape: {output_unfold_transpose.shape}, {output_unfold_transpose.is_contiguous()}')
print(f'[VERBOSE] col2im shape: {final.shape}, {final.is_contiguous()}')
print(f'[VERBOSE] max Diff (vs. torch.matmul): {diff}')

y2 = conv2d_cuda.forward(x, weight, padding, stride)
diff2 = (y - y2).abs().max()

print(f'[CUSTOM CUDA] output shape: {y2.shape}')
print(f'[CUSTOM CUDA] max Diff (vs. torch.matmul): {diff2}')
#print(y.size(3)) # tensor.size(i) returns the value of ith dimension