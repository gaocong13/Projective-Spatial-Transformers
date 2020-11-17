/* This is the main c++ code of our ProST grid generator function with CPU support only.
   Please cite our paper if you use this implementation: "Generalizing Spatial Transformers to Projective Geometry with Applications to 2D/3D Registration"
   @Copyright: Cong Gao, the Johns Hopkins University. Email: cgao11@jhu.edu
 */
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/extension.h>

#include <c10/util/ArrayRef.h>
#include <vector>
#include <math.h>

/* ProST base grid generation CPU function */
torch::Tensor ProST_grid_generator_5D_cpu_forward(
  const torch::Tensor &theta, // pose parameter
  int64_t N,        // batch size
  int64_t C,        // projection channel, currently only supports 1
  int64_t H,        // projection dim H
  int64_t W,        // projection dim W
  float dist_min,   // min distance from src to volume
  float dist_max,   // max distance from src to volume
  float src,        // source z-axis coordinate
  float det,        // detector plane z-axis coordinate
  float pix_spacing,// projection pixel spacing, currently only supports isotropic pixel
  float step_size,  // ray casting sampling step size
  bool align_corners)
{
  // Find source-to-detector distance
  float src_det_len = abs(src - det);
  // Create sampling range vector
  auto range = torch::range(dist_min-step_size, dist_max+step_size, step_size, theta.options());
  int64_t rc_pts = range.size(0);
  // Initialize ProST grid variable
  auto ProST_grid = torch::empty({N, H, W, rc_pts, 4}, theta.options());
  // basic range vectors for future reference
  auto tmp_range_x = range;
  auto tmp_range_y = range;
  auto tmp_range_z = range;

  float idx, idy, ray_len_idx, x_to_vec, y_to_vec, z_to_vec;
  // For loop over batch size
  for(int cntn = 0; cntn < N; cntn++){
    // For loop over each pixel on 2D projection
    for(int cntx = 0; cntx < H; cntx++){
      // Current pixel x-coordinate in projection plane
      idx = -pix_spacing*H/2 + cntx*pix_spacing;
      for(int cnty = 0; cnty < W; cnty++){
        // Current pixel y-coordinate in projection plane
        idy = -pix_spacing*W/2 + cnty*pix_spacing;
        // Length of ray-casting line connecting source to current pixel
        ray_len_idx = sqrt(idx*idx + idy*idy + src_det_len*src_det_len);
        // x, y, z to ray-casting line ratio
        x_to_vec = idx / ray_len_idx;
        y_to_vec = idy / ray_len_idx;
        z_to_vec = src_det_len / ray_len_idx;
        // Apply to basic range vector
        tmp_range_x = range * x_to_vec;
        tmp_range_y = range * y_to_vec;
        tmp_range_z = src - range * z_to_vec;
        // Fill in values
        ProST_grid.select(0, cntn).select(0, cntx).select(0, cnty).select(-1, 0).copy_(tmp_range_x);
        ProST_grid.select(0, cntn).select(0, cntx).select(0, cnty).select(-1, 1).copy_(tmp_range_y);
        ProST_grid.select(0, cntn).select(0, cntx).select(0, cnty).select(-1, 2).copy_(tmp_range_z);
        ProST_grid.select(0, cntn).select(0, cntx).select(0, cnty).select(-1, 3).fill_(1);
      }
    }
  }
  return ProST_grid.view({N, -1, 4});
}

torch::Tensor ProST_grid_generator_forward(const torch::Tensor &theta, torch::IntArrayRef size, float dist_min, float dist_max, float src, float det, float pix_spacing, float step_size, bool align_corners) {
  return ProST_grid_generator_5D_cpu_forward(theta, size[0], size[1], size[2], size[3], dist_min, dist_max, src, det, pix_spacing, step_size, align_corners);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &ProST_grid_generator_forward, "ProSTGrid Generator (CPU)");
}
