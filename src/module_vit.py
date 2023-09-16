from vit_pytorch.cross_vit import CrossViT
import torch.nn as nn
from util import _bilinear_interpolate_no_torch_5D
import torchgeometry as tgm
from posevec2mat import pose_vec2mat, inv_pose_vec, raydist_range
device = torch.device("cuda")

def transform_nan_check(dist_min, dist_max, transform_mat4x4, transform_mat3x4):
    if torch.isnan(dist_min).any() or torch.isnan(dist_max).any() \
                or torch.isnan(transform_mat4x4).any() or (torch.abs(transform_mat3x4) > 1000).any()\
                or (torch.abs(dist_min) > 1000).any() or (torch.abs(dist_max) > 1000).any():
        return False
    else:
        return True


class RegiNet_CrossViTv2_SW(nn.Module):
    def __init__(self):
        super(RegiNet_CrossViTv2_SW, self).__init__()

        # Define 3D convolutional layers        
        self._3D_conv = nn.Sequential(
            nn.Conv3d(1, 4, 3, 1, 1),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(4, 8, 3, 1, 1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 16, 3, 1, 1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(16, 8, 3, 1, 1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(8, 3, 3, 1, 1)
        )

        self._2Dconv_3x1 = nn.Conv2d(3, 1, 3, 1, 1)

        self._2Dconv_encode = CrossViT(
                                image_size = 128,
                                channels = 2,
                                num_classes = 1000,
                                depth = 3,
                                sm_dim = 16,            # high res dimension
                                sm_patch_size = 8,      # high res patch size (should be smaller than lg_patch_size)
                                sm_enc_depth = 2,        # high res depth
                                sm_enc_heads = 4,        # high res heads
                                sm_enc_mlp_dim = 256,   # high res feedforward dimension
                                lg_dim = 64,            # low res dimension
                                lg_patch_size = 64,      # low res patch size
                                lg_enc_depth = 3,        # low res depth
                                lg_enc_heads = 4,        # low res heads
                                lg_enc_mlp_dim = 256,   # low res feedforward dimensions
                                cross_attn_depth = 2,    # cross attention rounds
                                cross_attn_heads = 4,    # cross attention heads
                            )

        self._out_layer = nn.Linear(1000, 1)

    def forward(self, x, y, theta, corner_pt, param, log_nan_tensor=False):
        src = param[0]
        det = param[1]
        pix_spacing = param[2]
        step_size = param[3]

        x_exp = x.repeat(1,3,1,1,1)
        if self.no_3d_net:
            x_3d = x_exp
        else:
            x_3d = self._3D_conv(x) if self.no_3d_ori else x_exp + self._3D_conv(x)

        BATCH_SIZE = theta.size()[0]
        H = y.size()[2]
        W = y.size()[3]

        transform_mat4x4 = tgm.rtvec_to_pose(theta)
        transform_mat3x4 = transform_mat4x4[:, :3, :]
        dist_min, dist_max = raydist_range(transform_mat3x4, corner_pt, src)

        if not transform_nan_check(dist_min, dist_max, transform_mat4x4, transform_mat3x4):
            return False

        grid = ProSTGrid.forward(corner_pt, y.size(), dist_min.data, dist_max.data,
                                         src, det, pix_spacing, step_size, False)

        grid_trans = grid.bmm(transform_mat3x4.transpose(1,2)).view(BATCH_SIZE, H, W, -1, 3)

        x_3d = _bilinear_interpolate_no_torch_5D(x_3d, grid_trans)

        x_2d = torch.sum(x_3d, dim=-1)

        x_2d = self._2Dconv_3x1(x_2d)
        x_y_cat = torch.cat((x_2d, y), dim=1)

        out = self._2Dconv_encode(x_y_cat)
        out = self._out_layer(out)

        if torch.isnan(out).any():
            return False

        return out, True
