#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/extension.h>

#include <c10/util/ArrayRef.h>
#include <vector>
#include <math.h> 

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor ProST_grid_generator_5D_cuda_forward(
  const torch::Tensor &theta,
  int64_t N,
  int64_t C,
  int64_t H,
  int64_t W,
  float dist_min,
  float dist_max,
  float src,
  float det,
  float pix_spacing,
  float step_size,
  bool align_corners
  ); 

torch::Tensor ProST_grid_generator_forward(const torch::Tensor &theta, torch::IntArrayRef size, float dist_min, float dist_max, float src, float det, float pix_spacing, float step_size, bool align_corners) {
    return ProST_grid_generator_5D_cuda_forward(theta, size[0], size[1], size[2], size[3], dist_min, dist_max, src, det, pix_spacing, step_size, align_corners);
}
  
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &ProST_grid_generator_forward, "ProSTGrid Generator (CUDA)");
}  


