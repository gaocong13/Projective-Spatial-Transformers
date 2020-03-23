/* This is the CUDA kernel of our ProST grid generator function.
   Please cite our paper if you use this implementation: "Generalizing Spatial Transformers to Projective Geometry with Applications to 2D/3D Registration"
   @Copyright: Cong Gao, the Johns Hopkins University. Email: cgao11@jhu.edu
 */

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <math.h>

namespace{
  template <typename scalar_t>
  __global__ void make_ProST_grid_5D_kernel(
    torch::PackedTensorAccessor<scalar_t,5,torch::RestrictPtrTraits,size_t> ProST_grid,
    torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> range,
    int64_t N,
    int64_t H,
    int64_t W,
    float src,
    float pix_spacing,
    float src_det_len
    ){
      // row index
      const int r = blockIdx.x * blockDim.x + threadIdx.x;
      // column index
      const int c = blockIdx.y * blockDim.y + threadIdx.y;
      const auto inp_rc_pts = range.size(0);
      float ray_len_idx;
      float x_to_vec;
      float y_to_vec;
      float z_to_vec;
      float tmp_range_x;
      float tmp_range_y;
      float tmp_range_z;
      float idx, idy;
      if (r < ProST_grid.size(1) && c < ProST_grid.size(2)){
        idx = -pix_spacing*H/2 + r*pix_spacing;
        idy = -pix_spacing*W/2 + c*pix_spacing;
        ray_len_idx = sqrt(idx*idx + idy*idy + src_det_len*src_det_len);
        x_to_vec = idx / ray_len_idx;
        y_to_vec = idy / ray_len_idx;
        z_to_vec = src_det_len / ray_len_idx;
        for(size_t bn=0; bn<N; ++bn){
          for (size_t rc_ind = 0; rc_ind < inp_rc_pts; ++rc_ind){
            tmp_range_x = range[rc_ind] * x_to_vec;
            tmp_range_y = range[rc_ind] * y_to_vec;
            tmp_range_z = src - range[rc_ind] * z_to_vec;
            ProST_grid[bn][r][c][rc_ind][0] = tmp_range_x;
            ProST_grid[bn][r][c][rc_ind][1] = tmp_range_y;
            ProST_grid[bn][r][c][rc_ind][2] = tmp_range_z;
            ProST_grid[bn][r][c][rc_ind][3] = 1;
          }
        }
      }
    }
}

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
  ) {
    float src_det_len = abs(src - det);

    auto range = torch::range(dist_min-step_size, dist_max+step_size, step_size, theta.options());
    const auto rc_pts = range.size(0);
    auto ProST_grid = torch::empty({N, H, W, rc_pts, 4}, theta.options());
    auto tmp_range_x = range;
    auto tmp_range_y = range;
    auto tmp_range_z = range;
    // Pre-defined GPU block dimensions
    const int dimx = 32;
    const int dimy = 32;
    const dim3 dimBlock(dimx, dimy);
    const dim3 dimGrid((int)ceil(H/dimBlock.x), (int)ceil(W/dimBlock.y));

    AT_DISPATCH_FLOATING_TYPES(ProST_grid.type(), "ProST_grid_generator_5D_cuda_forward", ([&] {
    make_ProST_grid_5D_kernel<scalar_t><<<dimGrid, dimBlock>>>(
        ProST_grid.packed_accessor<scalar_t, 5, torch::RestrictPtrTraits,size_t>(),
        range.packed_accessor<scalar_t, 1, torch::RestrictPtrTraits,size_t>(),
        N,
        H,
        W,
        src,
        pix_spacing,
        src_det_len
      );
      }));
    return ProST_grid.view({N, -1, 4});
}
