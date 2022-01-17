/** C++ Extension Refereneces
  *
  * 1. Official Tutorial
  *        https://pytorch.org/tutorials/advanced/cpp_extension.html
  *
  * 2. at::Tensor vs. torch::Tensor
  *        https://discuss.pytorch.org/t/difference-between-torch-tensor-and-at-tensor/35806/5
  *        https://github.com/pytorch/pytorch/issues/14257#issuecomment-440487374
  *
  * 3. Default arguments
  *        https://github.com/pytorch/extension-cpp/issues/5
  * 
  * 4. C++ Docs
  *        https://pytorch.org/cppdocs/api/library_root.html
  */

#include <torch/extension.h> // includes ATen library, pybind11, interaction
#include <iostream>
#include <vector>
#include <omp.h>
#include <mkl.h>

#define PRINT_ENABLE false

/** Function Call Hierarchy 
  *
  * torch.nn.EmbeddingBag (torch/nn/modules/sparse.py) -> EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
  *                                                    -> V = E(sparse_index_group_batch,
  *                                                             sparse_offset_group_batch,
  *                                                             per_sample_weights=per_sample_weights)
  *
  *     F.embedding_bag (torch/nn/functional.py)       -> return F.embedding_bag(input, self.weight, offsets,
  *                                                                              self.max_norm, self.norm_type,
  *                                                                              self.scale_grad_by_freq, self.mode, self.sparse,
  *                                                                              per_sample_weights, self.include_last_offset,
  *                                                                              self.padding_idx)
  *         torch.embedding_bag
  */

/* Specification of the nn.EmbeddingBag forward operator */
/*
def forward(self, input: Tensor, offsets: Optional[Tensor] = None, per_sample_weights: Optional[Tensor] = None) -> Tensor:
        """Forward pass of EmbeddingBag.

        Args:
            input (Tensor): Tensor containing bags of indices into the embedding matrix.
            offsets (Tensor, optional): Only used when :attr:`input` is 1D. :attr:`offsets` determines
                the starting index position of each bag (sequence) in :attr:`input`.
            per_sample_weights (Tensor, optional): a tensor of float / double weights, or None
                to indicate all weights should be taken to be ``1``. If specified, :attr:`per_sample_weights`
                must have exactly the same shape as input and is treated as having the same
                :attr:`offsets`, if those are not ``None``. Only supported for ``mode='sum'``.

        Returns:
            Tensor output shape of `(B, embedding_dim)`.

        .. note::

            A few notes about ``input`` and ``offsets``:

            - :attr:`input` and :attr:`offsets` have to be of the same type, either int or long

            - If :attr:`input` is 2D of shape `(B, N)`, it will be treated as ``B`` bags (sequences)
              each of fixed length ``N``, and this will return ``B`` values aggregated in a way
              depending on the :attr:`mode`. :attr:`offsets` is ignored and required to be ``None`` in this case.

            - If :attr:`input` is 1D of shape `(N)`, it will be treated as a concatenation of
              multiple bags (sequences).  :attr:`offsets` is required to be a 1D tensor containing the
              starting index positions of each bag in :attr:`input`. Therefore, for :attr:`offsets` of shape `(B)`,
              :attr:`input` will be viewed as having ``B`` bags. Empty bags (i.e., having 0-length) will have
              returned vectors filled by zeros.
        """
        return F.embedding_bag(input, self.weight, offsets,
                               self.max_norm, self.norm_type,
                               self.scale_grad_by_freq, self.mode, self.sparse,
                               per_sample_weights, self.include_last_offset,
                               self.padding_idx)
*/

/* custom embedding reduction with per_sample_weights */
torch::Tensor embedding_reduction_sum(
  torch::Tensor input, // 1D Tensor containing bags of indices
  torch::Tensor weight,// embedding matrix/table of (N, M)
  torch::Tensor offsets, // 1D Tensor of starting index position of each bag (sequence) in input
  torch::Tensor per_sample_weights // 1D Tensor that represents weight of each bag (same dimension as input, follows offsets)
) {

  int batch = offsets.sizes()[0];
  int dim = weight.sizes()[1];
  int num_reduction = input.sizes()[0];

  torch::Tensor output = torch::zeros({batch, dim}); // (B , M)

  #if PRINT_ENABLE
  std::cout << "Batch Size: " << batch << std::endl;
  std::cout << "Embedding Vector Dimension: " << dim << std::endl;
  std::cout << "Total Number of Reduction: " << num_reduction << std::endl;
  std::cout << "Avg. Pooling Size: " << num_reduction / batch << std::endl;
  #endif

  int *input_ptr = input.data_ptr<int>();
  float *weight_ptr = weight.data_ptr<float>();
  int *offsets_ptr = offsets.data_ptr<int>();
  float *output_ptr = output.data_ptr<float>();
  float *per_sample_weights_ptr = per_sample_weights.data_ptr<float>();

  #pragma omp parallel for
  for (int i = 0; i < batch; i++) {
    for (int j = offsets_ptr[i]; j < (i == (batch-1) ? num_reduction : offsets_ptr[i+1]); j++) {
      // for (int k = 0; k < dim; k++) {
      //   output_ptr[dim*i + k] += per_sample_weights_ptr[j] * weight_ptr[dim*input_ptr[j] + k];
      // }
      cblas_saxpy(dim, per_sample_weights_ptr[j], weight_ptr + dim*input_ptr[j], 1, output_ptr + dim*i, 1);
    }
  }

  return output;
}

/* custom embedding reduction without per_sample_weights */
torch::Tensor embedding_reduction_sum2(
  torch::Tensor input, // 1D Tensor containing bags of indices
  torch::Tensor weight,// embedding matrix/table of (N, M)
  torch::Tensor offsets // 1D Tensor of starting index position of each bag (sequence) in input
) {

  int batch = offsets.sizes()[0];
  int dim = weight.sizes()[1];
  int num_reduction = input.sizes()[0];

  torch::Tensor output = torch::zeros({batch, dim}); // (B , M)

  #if PRINT_ENABLE
  std::cout << "Batch Size: " << batch << std::endl;
  std::cout << "Embedding Vector Dimension: " << dim << std::endl;
  std::cout << "Total Number of Reduction: " << num_reduction << std::endl;
  std::cout << "Avg. Pooling Size: " << num_reduction / batch << std::endl;
  #endif

  long *input_ptr = input.data_ptr<long>();
  float *weight_ptr = weight.data_ptr<float>();
  long *offsets_ptr = offsets.data_ptr<long>();
  float *output_ptr = output.data_ptr<float>();

  #pragma omp parallel for
  for (int i = 0; i < batch; i++) {
    for (int j = offsets_ptr[i]; j < (i == (batch-1) ? num_reduction : offsets_ptr[i+1]); j++) {
      // for (int k = 0; k < dim; k++) {
      //   output_ptr[dim*i + k] += weight_ptr[dim*input_ptr[j] + k];
      // }
      cblas_saxpy(dim, 1, weight_ptr + dim*input_ptr[j], 1, output_ptr + dim*i, 1);
    }
  }

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &embedding_reduction_sum, "custom embedding reduction with per_sample_weights");
  m.def("forward2", &embedding_reduction_sum2, "custom embedding reduction without per_sample_weights");
}
