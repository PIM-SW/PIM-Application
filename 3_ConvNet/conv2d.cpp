#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA Declarations
torch::Tensor conv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int padding,
    int stride);

// 2D Convolution without bias (i.e., bias=False)
torch::Tensor conv2d_forward(torch::Tensor input,
                            torch::Tensor weight,
                            int padding,
                            int stride) {
    
    // Check input tensors
    CHECK_INPUT(input);
    CHECK_INPUT(weight);

    // Call relaying function
    return conv2d_cuda_forward(input, weight, padding, stride);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv2d_forward, "Conv2D CUDA Kernel");
}