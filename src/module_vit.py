import torch
from vit_pytorch.vit_for_small_dataset import ViT, ViTv2
from vit_pytorch.cait import CaiT
from vit_pytorch.t2t import T2TViT
from vit_pytorch.cross_vit import CrossViT, CrossViTv3, CrossViTv4
from vit_pytorch.twins_svt import TwinsSVT

import ProSTGrid
import torch.nn as nn
import numpy as np
from util import _bilinear_interpolate_no_torch_5D
import torchgeometry as tgm
import torch.nn.functional as F
from posevec2mat import pose_vec2mat, inv_pose_vec, raydist_range
from resunet_3d import resunet_3d
import nibabel as nib

device = torch.device("cuda")

def norm_x_2d(x_2d, BATCH_SIZE, H, W):
    min_x2d, _ = torch.min(x_2d.reshape(BATCH_SIZE * 3, -1), dim=-1, keepdim=True)
    max_x2d, _ = torch.max(x_2d.reshape(BATCH_SIZE * 3, -1), dim=-1, keepdim=True)
    x_2d = (x_2d.reshape(BATCH_SIZE * 3, -1) - min_x2d) / (max_x2d - min_x2d)
    x_2d = x_2d.reshape(BATCH_SIZE, 3, H, W)

    return x_2d

def transform_nan_check(dist_min, dist_max, transform_mat4x4, transform_mat3x4):
    if torch.isnan(dist_min).any() or torch.isnan(dist_max).any() \
                or torch.isnan(transform_mat4x4).any() or (torch.abs(transform_mat3x4) > 1000).any()\
                or (torch.abs(dist_min) > 1000).any() or (torch.abs(dist_max) > 1000).any():
        return False
    else:
        return True

class RegiNet_smallViT(nn.Module):
    def __init__(self, SAVE_PATH, generate_proj_mov=False, norm_mov=False, no_3d_ori=False):
        super(RegiNet_smallViT, self).__init__()
        self.save_path = SAVE_PATH
        self.do_proj_mov = generate_proj_mov #generate proj_mov for debugging purpose
        self.norm_mov = norm_mov
        self.no_3d_ori = no_3d_ori

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

        self._2Dconv_encode_x = ViT(
                                    image_size = 128,
                                    channels = 3,
                                    patch_size = 16,
                                    num_classes = 1000,
                                    dim = 128,
                                    depth = 3,
                                    heads = 4,
                                    mlp_dim = 256
                                )

        self._2Dconv_encode_y = ViT(
                                    image_size = 128,
                                    channels = 3,
                                    patch_size = 16,
                                    num_classes = 1000,
                                    dim = 128,
                                    depth = 3,
                                    heads = 4,
                                    mlp_dim = 256
                                )

    def forward(self, x, y, theta, corner_pt, param, log_nan_tensor=False):
        src = param[0]
        det = param[1]
        pix_spacing = param[2]
        step_size = param[3]

        x_exp = x.repeat(1,3,1,1,1)
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

        if self.norm_mov:
            x_2d = norm_x_2d(x_2d, BATCH_SIZE, H, W)

        y_out = self._2Dconv_encode_y(y.repeat(1,3,1,1))
        x_out = self._2Dconv_encode_x(x_2d)

        if torch.isnan(y_out).any() or torch.isnan(x_out).any():
            return False

        if self.do_proj_mov:
            x_3d_ad = _bilinear_interpolate_no_torch_5D(x, grid_trans)
            x_2d_ad = torch.sum(x_3d_ad, dim=-1)

            return x_out, y_out, x_2d_ad, True

        return x_out, y_out, True

class RegiNet_smallViT_SW(nn.Module):
    def __init__(self, SAVE_PATH, generate_proj_mov=False, norm_mov=False, no_3d_ori=False):
        super(RegiNet_smallViT_SW, self).__init__()
        self.save_path = SAVE_PATH
        self.do_proj_mov = generate_proj_mov #generate proj_mov for debugging purpose
        self.norm_mov = norm_mov
        self.no_3d_ori = no_3d_ori

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

        self._2Dconv_3x1 = nn.Sequential(
            nn.Conv2d(3, 1, 3, 1, 1),
            nn.ReLU()
        )

        self._2Dconv_encode = ViT(
                                image_size = 128,
                                channels = 2,
                                patch_size = 16,
                                num_classes = 1000,
                                dim = 128,
                                depth = 3,
                                heads = 4,
                                mlp_dim = 256
                            )

        self._out_layer = nn.Linear(1000, 1)

    def forward(self, x, y, theta, corner_pt, param, log_nan_tensor=False):
        src = param[0]
        det = param[1]
        pix_spacing = param[2]
        step_size = param[3]

        x_exp = x.repeat(1,3,1,1,1)
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

        if self.norm_mov:
            x_2d = norm_x_2d(x_2d, BATCH_SIZE, H, W)

        x_2d = self._2Dconv_3x1(x_2d)
        x_y_cat = torch.cat((x_2d, y), dim=1)

        out = self._2Dconv_encode(x_y_cat)
        out = self._out_layer(out)

        if torch.isnan(out).any():
            return False

        if self.do_proj_mov:
            x_3d_ad = _bilinear_interpolate_no_torch_5D(x, grid_trans)
            x_2d_ad = torch.sum(x_3d_ad, dim=-1)

            return out, x_2d_ad, True

        return out, True

class RegiNet_smallViTv2_SW(nn.Module):
    def __init__(self, SAVE_PATH, generate_proj_mov=False, norm_mov=False, no_3d_ori=False):
        super(RegiNet_smallViTv2_SW, self).__init__()
        self.save_path = SAVE_PATH
        self.do_proj_mov = generate_proj_mov #generate proj_mov for debugging purpose
        self.norm_mov = norm_mov
        self.no_3d_ori = no_3d_ori

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

        self._2Dconv_3x1 = nn.Sequential(
            nn.Conv2d(3, 1, 3, 1, 1),
            nn.ReLU()
        )

        self._2Dconv_encode_x = ViTv2(
                                image_size = 128,
                                channels = 1,
                                patch_size = 16,
                                dim = 512,
                                depth = 6,
                                heads = 4,
                                mlp_dim = 1024
                            )

        self._2Dconv_encode_y = ViTv2(
                                image_size = 128,
                                channels = 1,
                                patch_size = 16,
                                dim = 512,
                                depth = 6,
                                heads = 4,
                                mlp_dim = 1024
                            )

        self._out_layer = nn.Sequential(nn.Linear(1024, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 16),
                                        nn.ReLU(),
                                        nn.Linear(16,1))

    def forward(self, x, y, theta, corner_pt, param, log_nan_tensor=False):
        src = param[0]
        det = param[1]
        pix_spacing = param[2]
        step_size = param[3]

        x_exp = x.repeat(1,3,1,1,1)
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

        if self.norm_mov:
            x_2d = norm_x_2d(x_2d, BATCH_SIZE, H, W)

        x_2d = self._2Dconv_3x1(x_2d)
        x_encode = self._2Dconv_encode_x(x_2d)
        y_encode = self._2Dconv_encode_y(y)

        x_y_cat = torch.cat((x_encode, y_encode), dim=1)
        out = self._out_layer(x_y_cat)

        if torch.isnan(out).any():
            return False

        if self.do_proj_mov:
            x_3d_ad = _bilinear_interpolate_no_torch_5D(x, grid_trans)
            x_2d_ad = torch.sum(x_3d_ad, dim=-1)

            return out, x_2d_ad, True

        return out, True

class RegiNet_CrossViT(nn.Module):
    def __init__(self, SAVE_PATH, generate_proj_mov=False, norm_mov=False, no_3d_ori=False):
        super(RegiNet_CrossViT, self).__init__()
        self.save_path = SAVE_PATH
        self.do_proj_mov = generate_proj_mov #generate proj_mov for debugging purpose
        self.norm_mov = norm_mov
        self.no_3d_ori = no_3d_ori

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

        self._2Dconv_encode_x = CrossViT(
                                    image_size = 128,
                                    channels = 3,
                                    num_classes = 256,
                                    depth = 4,
                                    sm_dim = 192,            # high res dimension
                                    sm_patch_size = 64,      # high res patch size (should be smaller than lg_patch_size)
                                    sm_enc_depth = 2,        # high res depth
                                    sm_enc_heads = 4,        # high res heads
                                    sm_enc_mlp_dim = 512,   # high res feedforward dimension
                                    lg_dim = 192,            # low res dimension
                                    lg_patch_size = 32,      # low res patch size
                                    lg_enc_depth = 2,        # low res depth
                                    lg_enc_heads = 4,        # low res heads
                                    lg_enc_mlp_dim = 512,   # low res feedforward dimensions
                                    cross_attn_depth = 2,    # cross attention rounds
                                    cross_attn_heads = 8,    # cross attention heads
                                )

        self._2Dconv_encode_y = CrossViT(
                                    image_size = 128,
                                    channels = 1,
                                    num_classes = 256,
                                    depth = 4,
                                    sm_dim = 192,            # high res dimension
                                    sm_patch_size = 64,      # high res patch size (should be smaller than lg_patch_size)
                                    sm_enc_depth = 2,        # high res depth
                                    sm_enc_heads = 4,        # high res heads
                                    sm_enc_mlp_dim = 512,   # high res feedforward dimension
                                    lg_dim = 192,            # low res dimension
                                    lg_patch_size = 32,      # low res patch size
                                    lg_enc_depth = 2,        # low res depth
                                    lg_enc_heads = 4,        # low res heads
                                    lg_enc_mlp_dim = 512,   # low res feedforward dimensions
                                    cross_attn_depth = 2,    # cross attention rounds
                                    cross_attn_heads = 8,    # cross attention heads
                                )

    def forward(self, x, y, theta, corner_pt, param, log_nan_tensor=False):
        src = param[0]
        det = param[1]
        pix_spacing = param[2]
        step_size = param[3]

        x_exp = x.repeat(1,3,1,1,1)
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

        if self.norm_mov:
            x_2d = norm_x_2d(x_2d, BATCH_SIZE, H, W)

        y_out = self._2Dconv_encode_y(y)
        x_out = self._2Dconv_encode_x(x_2d)

        if torch.isnan(y_out).any() or torch.isnan(x_out).any():
            return False

        if self.do_proj_mov:
            x_3d_ad = _bilinear_interpolate_no_torch_5D(x, grid_trans)
            x_2d_ad = torch.sum(x_3d_ad, dim=-1)

            return x_out, y_out, x_2d_ad, True

        return x_out, y_out, True

class RegiNet_CrossViT_SW(nn.Module):
    def __init__(self, SAVE_PATH, generate_proj_mov=False, norm_mov=False, no_3d_ori=False, no_3d_net=False):
        super(RegiNet_CrossViT_SW, self).__init__()
        self.save_path = SAVE_PATH
        self.do_proj_mov = generate_proj_mov #generate proj_mov for debugging purpose
        self.norm_mov = norm_mov
        self.no_3d_ori = no_3d_ori
        self.no_3d_net = no_3d_net

        if not self.no_3d_net:
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
                                lg_patch_size = 32,      # low res patch size
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

        if self.norm_mov:
            x_2d = norm_x_2d(x_2d, BATCH_SIZE, H, W)

        x_2d = self._2Dconv_3x1(x_2d)
        x_y_cat = torch.cat((x_2d, y), dim=1)

        out = self._2Dconv_encode(x_y_cat)
        out = self._out_layer(out)

        if torch.isnan(out).any():
            return False

        if self.do_proj_mov:
            x_3d_ad = _bilinear_interpolate_no_torch_5D(x, grid_trans)
            x_2d_ad = torch.sum(x_3d_ad, dim=-1)

            return out, x_2d_ad, True

        return out, True

class RegiNet_CrossViTv2_SW(nn.Module):
    def __init__(self, SAVE_PATH, generate_proj_mov=False, norm_mov=False, no_3d_ori=False, no_3d_net=False):
        super(RegiNet_CrossViTv2_SW, self).__init__()
        self.save_path = SAVE_PATH
        self.do_proj_mov = generate_proj_mov #generate proj_mov for debugging purpose
        self.norm_mov = norm_mov
        self.no_3d_ori = no_3d_ori
        self.no_3d_net = no_3d_net

        if not self.no_3d_net:
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

        if self.norm_mov:
            x_2d = norm_x_2d(x_2d, BATCH_SIZE, H, W)

        x_2d = self._2Dconv_3x1(x_2d)
        x_y_cat = torch.cat((x_2d, y), dim=1)

        out = self._2Dconv_encode(x_y_cat)
        out = self._out_layer(out)

        if torch.isnan(out).any():
            return False

        if self.do_proj_mov:
            x_3d_ad = _bilinear_interpolate_no_torch_5D(x, grid_trans)
            x_2d_ad = torch.sum(x_3d_ad, dim=-1)

            return out, x_2d_ad, True

        return out, True

class RegiNet_CrossViTv3_SW(nn.Module):
    def __init__(self, SAVE_PATH, generate_proj_mov=False, norm_mov=False, no_3d_ori=False, no_3d_net=False):
        super(RegiNet_CrossViTv3_SW, self).__init__()
        self.save_path = SAVE_PATH
        self.do_proj_mov = generate_proj_mov #generate proj_mov for debugging purpose
        self.norm_mov = norm_mov
        self.no_3d_ori = no_3d_ori
        self.no_3d_net = no_3d_net

        if not self.no_3d_net:
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

        self._2Dconv_encode = CrossViTv3(
                                image_size = 128,
                                channels = 1,
                                num_classes = 256,
                                depth = 4,
                                sm_dim = 192,            # high res dimension
                                sm_patch_size = 32,      # high res patch size (should be smaller than lg_patch_size)
                                sm_enc_depth = 2,        # high res depth
                                sm_enc_heads = 4,        # high res heads
                                sm_enc_mlp_dim = 512,   # high res feedforward dimension
                                lg_dim = 192,            # low res dimension
                                lg_patch_size = 32,      # low res patch size
                                lg_enc_depth = 2,        # low res depth
                                lg_enc_heads = 4,        # low res heads
                                lg_enc_mlp_dim = 512,   # low res feedforward dimensions
                                cross_attn_depth = 2,    # cross attention rounds
                                cross_attn_heads = 8,    # cross attention heads
                            )

        self._out_layer = nn.Linear(256, 1)

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

        if self.norm_mov:
            x_2d = norm_x_2d(x_2d, BATCH_SIZE, H, W)

        x_2d = self._2Dconv_3x1(x_2d)

        out = self._2Dconv_encode(x_2d, y)
        out = self._out_layer(out)

        if torch.isnan(out).any():
            return False

        if self.do_proj_mov:
            x_3d_ad = _bilinear_interpolate_no_torch_5D(x, grid_trans)
            x_2d_ad = torch.sum(x_3d_ad, dim=-1)

            return out, x_2d_ad, True

        return out, True

class RegiNet_CrossViTv3_SW_single3D(nn.Module):
    def __init__(self, SAVE_PATH, generate_proj_mov=False, norm_mov=False, no_3d_ori=False, no_3d_net=False):
        super(RegiNet_CrossViTv3_SW_single3D, self).__init__()
        self.save_path = SAVE_PATH
        self.do_proj_mov = generate_proj_mov #generate proj_mov for debugging purpose
        self.norm_mov = norm_mov
        self.no_3d_ori = no_3d_ori
        self.no_3d_net = no_3d_net

        if not self.no_3d_net:
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
                nn.Conv3d(8, 1, 3, 1, 1)
            )

        self._2Dconv_encode = CrossViTv3(
                                image_size = 128,
                                channels = 1,
                                num_classes = 256,
                                depth = 4,
                                sm_dim = 192,            # high res dimension
                                sm_patch_size = 32,      # high res patch size (should be smaller than lg_patch_size)
                                sm_enc_depth = 2,        # high res depth
                                sm_enc_heads = 4,        # high res heads
                                sm_enc_mlp_dim = 512,   # high res feedforward dimension
                                lg_dim = 192,            # low res dimension
                                lg_patch_size = 32,      # low res patch size
                                lg_enc_depth = 2,        # low res depth
                                lg_enc_heads = 4,        # low res heads
                                lg_enc_mlp_dim = 512,   # low res feedforward dimensions
                                cross_attn_depth = 2,    # cross attention rounds
                                cross_attn_heads = 8,    # cross attention heads
                            )

        self._out_layer = nn.Linear(256, 1)

    def forward(self, x, y, theta, corner_pt, param, log_nan_tensor=False):
        src = param[0]
        det = param[1]
        pix_spacing = param[2]
        step_size = param[3]

        if self.no_3d_net:
            x_3d = x
        else:
            x_3d = self._3D_conv(x) if self.no_3d_ori else x + self._3D_conv(x)

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

        if self.norm_mov:
            x_2d = norm_x_2d(x_2d, BATCH_SIZE, H, W)

        out = self._2Dconv_encode(x_2d, y)
        out = self._out_layer(out)

        if torch.isnan(out).any():
            return False

        if self.do_proj_mov:
            x_3d_ad = _bilinear_interpolate_no_torch_5D(x, grid_trans)
            x_2d_ad = torch.sum(x_3d_ad, dim=-1)

            return out, x_2d_ad, True

        return out, True

class RegiNet_CrossViTv4(nn.Module):
    def __init__(self, SAVE_PATH, generate_proj_mov=False, norm_mov=False, no_3d_ori=False, no_3d_net=False):
        super(RegiNet_CrossViTv4, self).__init__()
        self.save_path = SAVE_PATH
        self.do_proj_mov = generate_proj_mov #generate proj_mov for debugging purpose
        self.norm_mov = norm_mov
        self.no_3d_ori = no_3d_ori
        self.no_3d_net = no_3d_net

        if not self.no_3d_net:
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
                nn.Conv3d(8, 1, 3, 1, 1)
            )

        self._2Dconv_encode = CrossViTv4(
                                image_size = 128,
                                channels = 1,
                                num_classes = 256,
                                depth = 4,
                                sm_dim = 192,            # high res dimension
                                sm_patch_size = 32,      # high res patch size (should be smaller than lg_patch_size)
                                sm_enc_depth = 2,        # high res depth
                                sm_enc_heads = 4,        # high res heads
                                sm_enc_mlp_dim = 512,   # high res feedforward dimension
                                lg_dim = 192,            # low res dimension
                                lg_patch_size = 32,      # low res patch size
                                lg_enc_depth = 2,        # low res depth
                                lg_enc_heads = 4,        # low res heads
                                lg_enc_mlp_dim = 512,   # low res feedforward dimensions
                                cross_attn_depth = 2,    # cross attention rounds
                                cross_attn_heads = 8,    # cross attention heads
                            )

        self._out_layer = nn.Linear(192, 1)

    def forward(self, x, y, theta, corner_pt, param, log_nan_tensor=False):
        src = param[0]
        det = param[1]
        pix_spacing = param[2]
        step_size = param[3]

        if self.no_3d_net:
            x_3d = x
        else:
            x_3d = self._3D_conv(x) if self.no_3d_ori else x + self._3D_conv(x)

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

        if self.norm_mov:
            x_2d = norm_x_2d(x_2d, BATCH_SIZE, H, W)

        out = self._2Dconv_encode(x_2d, y)
        out = self._out_layer(out)

        if torch.isnan(out).any():
            return False

        if self.do_proj_mov:
            x_3d_ad = _bilinear_interpolate_no_torch_5D(x, grid_trans)
            x_2d_ad = torch.sum(x_3d_ad, dim=-1)

            return out, x_2d_ad, True

        return out, True

class RegiNet_CrossViTv5(nn.Module):
    def __init__(self, SAVE_PATH, generate_proj_mov=False, norm_mov=False, no_3d_ori=False, no_3d_net=False):
        super(RegiNet_CrossViTv5, self).__init__()
        self.save_path = SAVE_PATH
        self.do_proj_mov = generate_proj_mov #generate proj_mov for debugging purpose
        self.norm_mov = norm_mov
        self.no_3d_ori = no_3d_ori
        self.no_3d_net = no_3d_net

        if not self.no_3d_net:
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
                nn.Conv3d(8, 1, 3, 1, 1)
            )

        self._2Dconv_encode = CrossViTv4(
                                image_size = 128,
                                channels = 1,
                                num_classes = 256,
                                depth = 4,
                                sm_dim = 384,            # high res dimension
                                sm_patch_size = 64,      # high res patch size (should be smaller than lg_patch_size)
                                sm_enc_depth = 2,        # high res depth
                                sm_enc_heads = 4,        # high res heads
                                sm_enc_mlp_dim = 512,   # high res feedforward dimension
                                lg_dim = 384,            # low res dimension
                                lg_patch_size = 64,      # low res patch size
                                lg_enc_depth = 2,        # low res depth
                                lg_enc_heads = 4,        # low res heads
                                lg_enc_mlp_dim = 512,   # low res feedforward dimensions
                                cross_attn_depth = 2,    # cross attention rounds
                                cross_attn_heads = 8,    # cross attention heads
                                dropout = 0.0,
                                emb_dropout = 0.0
                            )

        self._out_layer = nn.Linear(384, 1)

    def forward(self, x, y, theta, corner_pt, param, log_nan_tensor=False):
        src = param[0]
        det = param[1]
        pix_spacing = param[2]
        step_size = param[3]

        if self.no_3d_net:
            x_3d = x
        else:
            x_3d = self._3D_conv(x) if self.no_3d_ori else x + self._3D_conv(x)

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

        if self.norm_mov:
            x_2d = norm_x_2d(x_2d, BATCH_SIZE, H, W)

        out = self._2Dconv_encode(x_2d, y)
        out = self._out_layer(out)

        if torch.isnan(out).any():
            return False

        if self.do_proj_mov:
            x_3d_ad = _bilinear_interpolate_no_torch_5D(x, grid_trans)
            x_2d_ad = torch.sum(x_3d_ad, dim=-1)

            return out, x_2d_ad, True

        return out, True

class RegiNet_CrossViTv3_SW_3Dresunet(nn.Module):
    def __init__(self, SAVE_PATH, generate_proj_mov=False, norm_mov=False, no_3d_ori=False, no_3d_net=False):
        super(RegiNet_CrossViTv3_SW_3Dresunet, self).__init__()
        self.save_path = SAVE_PATH
        self.do_proj_mov = generate_proj_mov #generate proj_mov for debugging purpose
        self.norm_mov = norm_mov
        self.no_3d_ori = no_3d_ori
        self.no_3d_net = no_3d_net

        if not self.no_3d_net:
            self._3D_conv = resunet_3d(1, 16, 1)

        self._2Dconv_encode = CrossViTv3(
                                image_size = 128,
                                channels = 1,
                                num_classes = 256,
                                depth = 4,
                                sm_dim = 192,            # high res dimension
                                sm_patch_size = 32,      # high res patch size (should be smaller than lg_patch_size)
                                sm_enc_depth = 2,        # high res depth
                                sm_enc_heads = 4,        # high res heads
                                sm_enc_mlp_dim = 512,   # high res feedforward dimension
                                lg_dim = 192,            # low res dimension
                                lg_patch_size = 32,      # low res patch size
                                lg_enc_depth = 2,        # low res depth
                                lg_enc_heads = 4,        # low res heads
                                lg_enc_mlp_dim = 512,   # low res feedforward dimensions
                                cross_attn_depth = 2,    # cross attention rounds
                                cross_attn_heads = 8,    # cross attention heads
                            )

        self._out_layer = nn.Linear(256, 1)

    def forward(self, x, y, theta, corner_pt, param, log_nan_tensor=False):
        src = param[0]
        det = param[1]
        pix_spacing = param[2]
        step_size = param[3]

        if self.no_3d_net:
            x_3d = x
        else:
            x_3d = self._3D_conv(x)

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

        if self.norm_mov:
            x_2d = norm_x_2d(x_2d, BATCH_SIZE, H, W)

        out = self._2Dconv_encode(x_2d, y)
        out = self._out_layer(out)

        if torch.isnan(out).any():
            return False

        if self.do_proj_mov:
            x_3d_ad = _bilinear_interpolate_no_torch_5D(x, grid_trans)
            x_2d_ad = torch.sum(x_3d_ad, dim=-1)

            return out, x_2d_ad, True

        return out, True

class RegiNet_CaiT(nn.Module):
    def __init__(self, SAVE_PATH, generate_proj_mov=False, norm_mov=False, no_3d_ori=False):
        super(RegiNet_CaiT, self).__init__()
        self.save_path = SAVE_PATH
        self.do_proj_mov = generate_proj_mov #generate proj_mov for debugging purpose
        self.norm_mov = norm_mov
        self.no_3d_ori = no_3d_ori

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

        self._2Dconv_encode_x = CaiT(
                                    image_size = 128,
                                    patch_size = 16,
                                    num_classes = 1000,
                                    dim = 128,
                                    depth = 3,
                                    cls_depth = 2,
                                    heads = 4,
                                    mlp_dim = 256
                                )

        self._2Dconv_encode_y = CaiT(
                                    image_size = 128,
                                    patch_size = 16,
                                    num_classes = 1000,
                                    dim = 128,
                                    depth = 3,
                                    cls_depth = 2,
                                    heads = 4,
                                    mlp_dim = 256
                                )

    def forward(self, x, y, theta, corner_pt, param, log_nan_tensor=False):
        src = param[0]
        det = param[1]
        pix_spacing = param[2]
        step_size = param[3]

        x_exp = x.repeat(1,3,1,1,1)
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

        if self.norm_mov:
            x_2d = norm_x_2d(x_2d, BATCH_SIZE, H, W)

        y_out = self._2Dconv_encode_y(y.repeat(1,3,1,1))
        x_out = self._2Dconv_encode_x(x_2d)

        if torch.isnan(y_out).any() or torch.isnan(x_out).any():
            return False

        if self.do_proj_mov:
            x_3d_ad = _bilinear_interpolate_no_torch_5D(x, grid_trans)
            x_2d_ad = torch.sum(x_3d_ad, dim=-1)

            return x_out, y_out, x_2d_ad, True

        return x_out, y_out, True

class RegiNet_T2TViT(nn.Module):
    def __init__(self, SAVE_PATH, generate_proj_mov=False, norm_mov=False, no_3d_ori=False):
        super(RegiNet_T2TViT, self).__init__()
        self.save_path = SAVE_PATH
        self.do_proj_mov = generate_proj_mov #generate proj_mov for debugging purpose
        self.norm_mov = norm_mov
        self.no_3d_ori = no_3d_ori

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

        self._2Dconv_encode_x = T2TViT(
                                    image_size = 128,
                                    num_classes = 1000,
                                    channels = 3,
                                    dim = 128,
                                    depth = 3,
                                    heads = 4,
                                    mlp_dim = 256,
                                    t2t_layers = ((7, 4), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
                                )

        self._2Dconv_encode_y = T2TViT(
                                    image_size = 128,
                                    num_classes = 1000,
                                    channels = 3,
                                    dim = 128,
                                    depth = 3,
                                    heads = 4,
                                    mlp_dim = 256,
                                    t2t_layers = ((7, 4), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
                                )

    def forward(self, x, y, theta, corner_pt, param, log_nan_tensor=False):
        src = param[0]
        det = param[1]
        pix_spacing = param[2]
        step_size = param[3]

        x_exp = x.repeat(1,3,1,1,1)
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

        if self.norm_mov:
            x_2d = norm_x_2d(x_2d, BATCH_SIZE, H, W)

        y_out = self._2Dconv_encode_y(y.repeat(1,3,1,1))
        x_out = self._2Dconv_encode_x(x_2d)

        if torch.isnan(y_out).any() or torch.isnan(x_out).any():
            return False

        if self.do_proj_mov:
            x_3d_ad = _bilinear_interpolate_no_torch_5D(x, grid_trans)
            x_2d_ad = torch.sum(x_3d_ad, dim=-1)

            return x_out, y_out, x_2d_ad, True

        return x_out, y_out, True

class RegiNet_TwinsSVT(nn.Module):
    def __init__(self, SAVE_PATH, generate_proj_mov=False, norm_mov=False, no_3d_ori=False):
        super(RegiNet_TwinsSVT, self).__init__()
        self.save_path = SAVE_PATH
        self.do_proj_mov = generate_proj_mov #generate proj_mov for debugging purpose
        self.norm_mov = norm_mov
        self.no_3d_ori = no_3d_ori

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

        self._2Dconv_encode_x = TwinsSVT(
                                    num_classes = 1000,       # number of output classes
                                    s1_emb_dim = 64,          # stage 1 - patch embedding projected dimension
                                    s1_patch_size = 4,        # stage 1 - patch size for patch embedding
                                    s1_local_patch_size = 7,  # stage 1 - patch size for local attention
                                    s1_global_k = 7,          # stage 1 - global attention key / value reduction factor, defaults to 7 as specified in paper
                                    s1_depth = 1,             # stage 1 - number of transformer blocks (local attn -> ff -> global attn -> ff)
                                    s2_emb_dim = 128,         # stage 2 (same as above)
                                    s2_patch_size = 2,
                                    s2_local_patch_size = 7,
                                    s2_global_k = 7,
                                    s2_depth = 1,
                                    s3_emb_dim = 256,         # stage 3 (same as above)
                                    s3_patch_size = 2,
                                    s3_local_patch_size = 7,
                                    s3_global_k = 7,
                                    s3_depth = 5,
                                    s4_emb_dim = 512,         # stage 4 (same as above)
                                    s4_patch_size = 2,
                                    s4_local_patch_size = 7,
                                    s4_global_k = 7,
                                    s4_depth = 4,
                                    peg_kernel_size = 3,      # positional encoding generator kernel size
                                )

        self._2Dconv_encode_y = TwinsSVT(
                                    num_classes = 1000,       # number of output classes
                                    s1_emb_dim = 64,          # stage 1 - patch embedding projected dimension
                                    s1_patch_size = 4,        # stage 1 - patch size for patch embedding
                                    s1_local_patch_size = 7,  # stage 1 - patch size for local attention
                                    s1_global_k = 7,          # stage 1 - global attention key / value reduction factor, defaults to 7 as specified in paper
                                    s1_depth = 1,             # stage 1 - number of transformer blocks (local attn -> ff -> global attn -> ff)
                                    s2_emb_dim = 128,         # stage 2 (same as above)
                                    s2_patch_size = 2,
                                    s2_local_patch_size = 7,
                                    s2_global_k = 7,
                                    s2_depth = 1,
                                    s3_emb_dim = 256,         # stage 3 (same as above)
                                    s3_patch_size = 2,
                                    s3_local_patch_size = 7,
                                    s3_global_k = 7,
                                    s3_depth = 5,
                                    s4_emb_dim = 512,         # stage 4 (same as above)
                                    s4_patch_size = 2,
                                    s4_local_patch_size = 7,
                                    s4_global_k = 7,
                                    s4_depth = 4,
                                    peg_kernel_size = 3,      # positional encoding generator kernel size
                                )

    def forward(self, x, y, theta, corner_pt, param, log_nan_tensor=False):
        src = param[0]
        det = param[1]
        pix_spacing = param[2]
        step_size = param[3]

        x_exp = x.repeat(1,3,1,1,1)
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

        if self.norm_mov:
            x_2d = norm_x_2d(x_2d, BATCH_SIZE, H, W)

        y_out = self._2Dconv_encode_y(y.repeat(1,3,1,1))
        x_out = self._2Dconv_encode_x(x_2d)

        if torch.isnan(y_out).any() or torch.isnan(x_out).any():
            return False

        if self.do_proj_mov:
            x_3d_ad = _bilinear_interpolate_no_torch_5D(x, grid_trans)
            x_2d_ad = torch.sum(x_3d_ad, dim=-1)

            return x_out, y_out, x_2d_ad, True

        return x_out, y_out, True
