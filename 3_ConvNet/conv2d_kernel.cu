#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <omp.h>
#include "cblas.h"
#include "pim_avail_op.h"

// CUDA Kernel
// Matrix multiplcation (A x B x C) x (C x D) = (A x B x D)
template <typename scalar_t>
__global__ void matmul_cuda(
    scalar_t* input,
    scalar_t* weight,
    scalar_t* output,
    int A,
    int B,
    int C,
    int D
) {

  // retrieve global index of each thread in the grid
  int i = blockDim.x * blockIdx.x + threadIdx.x; // A
  int j = blockDim.y * blockIdx.y + threadIdx.y; // B
  int k = blockDim.z * blockIdx.z + threadIdx.z; // D
  if (i >= A || j >= B || k >= D) return; // do not compute if the indices are out of range

  scalar_t sum = 0.0;
  for (int c = 0; c < C; c++) {
    sum += input[i*B*C + j*C + c] * weight[c*D + k]; // output(i, j, k)
  }
  output[i*B*D + j*D + k] = sum;
}

// Intermediate function
// Convolution implmented using im2col (i.e., convolution = unfold + matmul + fold)
torch::Tensor conv2d_cuda_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int padding,
    int stride) {

    /* Tensor dimensions */
    // Input: i_b x i_c x i_w x i_h
    // cf. tensor.size(i) == tensor.sizes()[i]
    int i_b = input.size(0); // batch
    int i_c = input.size(1); // in_channel
    int i_w = input.size(2); // width
    int i_h = input.size(3); // height
    // Weight: o_c x i_c x k_w x k_h
    int o_c = weight.size(0); // out_channel
    int k_w = weight.size(2); // kernel_width
    int k_h = weight.size(3); // kernel_height
    // Output: i_b x o_c x o_w x o_h
    int o_w = static_cast <int>(floor(((i_w - k_w + 2*padding) / stride) + 1)); // output_width
    int o_h = static_cast <int>(floor(((i_h - k_h + 2*padding) / stride) + 1)); // output_height

    // Output Tensors
    // [SSH] Temporarily commented out for CPU execution
    // auto options = torch::TensorOptions().device(torch::kCUDA); // default: dtype(kFloat32), layout(kStrided), device(kCPU), requires_grad(false)
    // torch::Tensor output = torch::zeros({i_b, o_w*o_h, o_c}, options).contiguous(); // implicit construction of TensorOptions from individual values
    torch::Tensor output = torch::zeros({i_b, o_w*o_h, o_c}).contiguous(); // implicit construction of TensorOptions from individual values

    // Modify input for matmul
    // i_b x i_c x i_w x i_h -> i_b x (i_c*k_w*k_h) x (o_w*o_h) -> i_b x (o_w*o_h) x (i_c*k_w*k_h) 
    torch::Tensor input_unfolded = torch::nn::functional::unfold(input, torch::nn::functional::UnfoldFuncOptions({k_w, k_h}).padding(padding).stride(stride)); // Apply im2Col
    torch::Tensor input_transposed = input_unfolded.transpose(1, 2).contiguous();

    // Reshape weight for matmul
    // o_c x i_c x k_w x k_h -> o_c x (i_c*k_w*k_h) -> (i_c*k_w*k_h) x o_c
    torch::Tensor weight_reshaped = weight.view({weight.size(0), -1});
    torch::Tensor weight_transposed = weight_reshaped.transpose(0, 1).contiguous();
    
    // CUDA Thread block
    // cf. each thread block is assigned to a SM
    // [SSH] Temporarily commented out for CPU execution
    // dim3 blockDim(1, 32, 32); // dimension of threads a single thread block
    // dim3 gridDim(i_b, static_cast<int>(ceil(o_w*o_h/32)), static_cast<int>(ceil(o_c/32))); // dimension of blocks in the grid

    // [SSH] Temporarily commented out for CPU execution
    // CUDA Kernel Execution
    // cf. macro that determines the type of tensor at runtime and selectively call functions with the corresponding type
    //     CUDA kernels are implemented using template in order to use this feature
    // AT_DISPATCH_FLOATING_TYPES(input.type(), "matmul_cuda", ([&] { // type, name (for error msg), lambda function
    //     matmul_cuda<scalar_t><<<gridDim, blockDim>>>(
    //         input_transposed.data_ptr<scalar_t>(), // according to PyTorch, Tensor.data<T>() is deprecated. Thus, we use Tensor.data_ptr<T>() instead.
    //         weight_transposed.data_ptr<scalar_t>(),
    //         output.data_ptr<scalar_t>(),
    //         i_b,
    //         o_w*o_h,
    //         i_c*k_w*k_h,
    //         o_c);
    // })); // i_b x (o_w*o_h) x (i_c*k_w*k_h) (matmul) (i_c*k_w*k_h) x o_c = i_b x (o_w*o_h) x o_c
    // torch::Tensor output_ref = torch::matmul(input_transposed, weight_transposed); // equivalent to torch::matmul

    #pragma omp parallel for
    for (int i=0; i<i_b; i++) {
      // [SSH] Apple format
      //     lda/ldb/ldc: The size of the first dimention of matrix A/B/C; if you are passing a matrix A/B/C[m][n], the value should be m.
      pimblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, o_w*o_h, o_c, i_c*k_w*k_h,
                    1.0, (input_transposed.data_ptr<float>() + i*(o_w*o_h)*(i_c*k_w*k_h)), o_w*o_h,
                    weight_transposed.data_ptr<float>(), i_c*k_w*k_h, 0.0,
                    (output.data_ptr<float>()+ i*(o_w*o_h)*o_c), o_w*o_h);

      // [SSH] Intel OneAPI MKL format
      //     lda: (NoTrans, Col): m, (NoTrans, Row): k, (Trans, Col): k, (Trans, Row): m
      //     ldb: (NoTrans, Col): k, (NoTrans, Row): n, (Trans, Col): n, (Trans, Row): k
      //     ldc: (Col): m, (Row): n
      // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, o_w*o_h, o_c, i_c*k_w*k_h,
      //               1.0, (input_transposed.data_ptr<float>() + i*(o_w*o_h)*(i_c*k_w*k_h)), i_c*k_w*k_h,
      //               weight_transposed.data_ptr<float>(), o_c, 0.0,
      //               (output.data_ptr<float>()+ i*(o_w*o_h)*o_c), o_c);
    }

    // i_b x (o_w*o_h) x o_c -> i_b x o_c x (o_w*o_h) x  -> i_b x o_c x o_w x o_h
    torch::Tensor output_transposed = output.transpose(1, 2); // Reshape the output for Col2im
    torch::Tensor output_folded = torch::nn::functional::fold(output_transposed, torch::nn::functional::FoldFuncOptions({o_w, o_h}, {1, 1})); // Apply Col2im

    return output_folded;
}