import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import math
import torchgeometry as tgm
import nibabel as nib
import cv2

from posevec2mat import euler2mat

PI = 3.1415926
criterion = nn.MSELoss()

def hounsfield2linearatten(vol):
    vol = vol.astype(float)
    mu_water_ = 0.02683*1.0
    mu_air_   = 0.02485*0.001
    hu_lower_ = -1000
    hu_scale_ = (mu_water_ - mu_air_) * 0.001
    mu_lower_ = (hu_lower_ * hu_scale_) + mu_water_
    for x in np.nditer(vol, op_flags=['readwrite']):
        x[...] = np.maximum((x*hu_scale_)+mu_water_-mu_lower_, 0.0)

    return vol

# Convert CT HU value to attenuation line integral
def conv_hu_to_density(vol):
    vol = vol.astype(float)
    mu_water_ = 0.02683*1.0
    mu_air_   = 0.02485*0.0001
    hu_lower_ = -130
    hu_scale_ = (mu_water_ - mu_air_) * 0.001
    mu_lower_ = (hu_lower_ * hu_scale_) + mu_water_
    densities = np.maximum((vol*hu_scale_) + mu_water_ - mu_lower_, 0)
    return densities

def tensor_exp2torch(T, BATCH_SIZE, device):
    T = np.expand_dims(T, axis=0)
    T = np.expand_dims(T, axis=0)
    T = np.repeat(T, BATCH_SIZE, axis=0)

    T = torch.tensor(T, dtype=torch.float, requires_grad=True, device=device)

    return T

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


'''
Defines ProST canonical geometries
input:
    CT_PATH, SEG_PATH: file path of CT and segmentation
    vol_spacing: needs to be calculated offline
    ISFlip: True if Z(IS) is flipped
output:
           param: src, det, pix_spacing, step_size, det_size
         _3D_vol: volume used for training DeepNet, we use CT segmentation
          CT_vol: CT
    ray_proj_mov: detector plane variable
       corner_pt: 8 corner points of input volume
     norm_factor: translation normalization factor
'''
def input_param(CT_PATH, SEG_PATH, BATCH_SIZE, vol_spacing = 2.33203125, ISFlip = False, device='cuda'):
    CT_vol = nib.load(CT_PATH)
    _3D_vol = nib.load(SEG_PATH)
    CT_vol = CT_vol.get_data()
    _3D_vol = _3D_vol.get_data()

    # Rotation 90 degrees for making an AP view projection
    CT_vol = np.rot90(CT_vol, 3)
    _3D_vol = np.rot90(_3D_vol, 3)

    CT_vol = conv_hu_to_density(CT_vol)
    _3D_vol = CT_vol * (_3D_vol>0)

    if ISFlip:
        CT_vol = np.flip(CT_vol, axis=2)
        _3D_vol = np.flip(_3D_vol, axis=2)

    # Pre-defined hard coded geometry
    src_det = 1020
    iso_center = 400
    det_size = 128
    pix_spacing = 0.73 * 512 / det_size #0.194*1536 / det_size
    step_size = 1.75
    vol_size = CT_vol.shape[0]

    norm_factor = (vol_size * vol_spacing / 2)
    src = (src_det - iso_center) / norm_factor
    det = -iso_center / norm_factor
    pix_spacing = pix_spacing / norm_factor
    step_size = step_size / norm_factor

    param = [src, det, pix_spacing, step_size, det_size]

    CT_vol = tensor_exp2torch(CT_vol, BATCH_SIZE, device)
    _3D_vol = tensor_exp2torch(_3D_vol, BATCH_SIZE, device)
    corner_pt = create_cornerpt(BATCH_SIZE, device)
    ray_proj_mov = np.zeros((det_size, det_size))
    ray_proj_mov = tensor_exp2torch(ray_proj_mov, BATCH_SIZE, device)

    return param, det_size, _3D_vol, CT_vol, ray_proj_mov, corner_pt, norm_factor


def init_rtvec_train(BATCH_SIZE, device):
     rtvec_gt = np.random.normal(0, 0.15, (BATCH_SIZE, 6))
     rtvec_gt[:, :3] = rtvec_gt[:, :3] * 0.35 * PI

     rtvec_smp = np.random.normal(0, 0.15, (BATCH_SIZE, 6))
     rtvec_smp[:, :3] = rtvec_smp[:, :3] * 0.35 * PI

     rtvec = rtvec_smp

     rtvec_torch = torch.tensor(rtvec, dtype=torch.float, requires_grad=True, device=device)
     rtvec_gt_torch = torch.tensor(rtvec_gt, dtype=torch.float, requires_grad=True, device=device)

     rot_mat = euler2mat(rtvec_torch[:, :3])
     angle_axis = tgm.rotation_matrix_to_angle_axis(torch.cat([rot_mat,  torch.zeros(BATCH_SIZE, 3, 1).to(device)], dim=-1))
     rtvec = torch.cat([angle_axis, rtvec_torch[:, 3:]], dim=-1)

     rot_mat_gt = euler2mat(rtvec_gt_torch[:, :3])
     angle_axis_gt = tgm.rotation_matrix_to_angle_axis(torch.cat([rot_mat_gt,  torch.zeros(BATCH_SIZE, 3, 1).to(device)], dim=-1))
     rtvec_gt = torch.cat([angle_axis_gt, rtvec_gt_torch[:, 3:]], dim=-1)
     transform_mat4x4_gt = tgm.rtvec_to_pose(rtvec_gt)
     transform_mat3x4_gt = transform_mat4x4_gt[:, :3, :]

     return transform_mat3x4_gt, rtvec, rtvec_gt


def init_rtvec_test(device, manual_test=False, manual_rtvec_gt=None, manual_rtvec_smp=None):
     BATCH_SIZE=1
     rtvec_gt = np.random.normal(0, 0.15, (BATCH_SIZE, 6))
     rtvec_gt[:, :3] = rtvec_gt[:, :3] * 0.35 * PI

     rtvec_smp = np.random.normal(0, 0.15, (BATCH_SIZE, 6))
     rtvec_smp[:, :3] = rtvec_smp[:, :3] * 0.3 * PI

     if manual_test:
         rtvec_gt = manual_rtvec_gt.copy()
         rtvec_smp = manual_rtvec_smp.copy()

     rtvec = rtvec_smp

     rtvec_torch = torch.tensor(rtvec, dtype=torch.float, requires_grad=True, device=device)
     rtvec_gt_torch = torch.tensor(rtvec_gt, dtype=torch.float, requires_grad=True, device=device)

     rot_mat = euler2mat(rtvec_torch[:, :3])
     angle_axis = tgm.rotation_matrix_to_angle_axis(torch.cat([rot_mat,  torch.zeros(BATCH_SIZE, 3, 1).to(device)], dim=-1))
     rtvec = torch.cat([angle_axis, rtvec_torch[:, 3:]], dim=-1)
     rtvec = torch.tensor(rtvec.detach(), requires_grad=True, device=device)

     rot_mat_gt = euler2mat(rtvec_gt_torch[:, :3])
     angle_axis_gt = tgm.rotation_matrix_to_angle_axis(torch.cat([rot_mat_gt,  torch.zeros(BATCH_SIZE, 3, 1).to(device)], dim=-1))
     rtvec_gt = torch.cat([angle_axis_gt, rtvec_gt_torch[:, 3:]], dim=-1)
     transform_mat4x4_gt = tgm.rtvec_to_pose(rtvec_gt)
     transform_mat3x4_gt = transform_mat4x4_gt[:, :3, :]

     return transform_mat3x4_gt, rtvec, rtvec_gt


def create_cornerpt(BATCH_SIZE, device):
    corner_pt = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]])
    corner_pt = torch.tensor(corner_pt.astype(float), requires_grad = False).type(torch.FloatTensor)
    corner_pt = corner_pt.unsqueeze(0).to(device)
    corner_pt = corner_pt.repeat(BATCH_SIZE, 1, 1)

    return corner_pt


def _repeat(x, n_repeats):
    with torch.no_grad():
        rep = torch.ones((1, n_repeats), dtype=torch.float32).cuda()

    return torch.matmul(x.view(-1, 1), rep).view(-1)


def _bilinear_interpolate_no_torch_5D(vol, grid):
    # Assume CT to be Nx1xDxHxW
    num_batch, channels, depth, height, width = vol.shape
    vol = vol.permute(0, 2, 3, 4, 1)
    _, out_depth, out_height, out_width, _ = grid.shape
    x =  width * (grid[:, :, :, :, 0] * 0.5 + 0.5)
    y = height * (grid[:, :, :, :, 1] * 0.5 + 0.5)
    z =  depth * (grid[:, :, :, :, 2] * 0.5 + 0.5)

    x = x.view(-1)
    y = y.view(-1)
    z = z.view(-1)

    ind = ~((x>=0) * (x<=width) * (y>=0) * (y<=height) * (z>=0) * (z<=depth))
    # do sampling
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1
    z0 = torch.floor(z)
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, width - 1)
    x1 = torch.clamp(x1, 0, width - 1)
    y0 = torch.clamp(y0, 0, height - 1)
    y1 = torch.clamp(y1, 0, height - 1)
    z0 = torch.clamp(z0, 0, depth - 1)
    z1 = torch.clamp(z1, 0, depth - 1)

    dim3 = float(width)
    dim2 = float(width * height)
    dim1 = float(depth * width * height)
    dim1_out = float(out_depth * out_width * out_height)

    base = _repeat(torch.arange(start=0, end=num_batch, dtype=torch.float32).cuda() * dim1, np.int32(dim1_out))
    idx_a = base.long() + (z0*dim2).long() + (y0*dim3).long() + x0.long()
    idx_b = base.long() + (z0*dim2).long() + (y0*dim3).long() + x1.long()
    idx_c = base.long() + (z0*dim2).long() + (y1*dim3).long() + x0.long()
    idx_d = base.long() + (z0*dim2).long() + (y1*dim3).long() + x1.long()
    idx_e = base.long() + (z1*dim2).long() + (y0*dim3).long() + x0.long()
    idx_f = base.long() + (z1*dim2).long() + (y0*dim3).long() + x1.long()
    idx_g = base.long() + (z1*dim2).long() + (y1*dim3).long() + x0.long()
    idx_h = base.long() + (z1*dim2).long() + (y1*dim3).long() + x1.long()

    # use indices to lookup pixels in the flat image and keep channels dim
    im_flat = vol.contiguous().view(-1, channels)
    Ia = im_flat[idx_a].view(-1, channels)
    Ib = im_flat[idx_b].view(-1, channels)
    Ic = im_flat[idx_c].view(-1, channels)
    Id = im_flat[idx_d].view(-1, channels)
    Ie = im_flat[idx_e].view(-1, channels)
    If = im_flat[idx_f].view(-1, channels)
    Ig = im_flat[idx_g].view(-1, channels)
    Ih = im_flat[idx_h].view(-1, channels)

    wa = torch.mul(torch.mul(x1 - x, y1 - y), z1 - z).view(-1, 1)
    wb = torch.mul(torch.mul(x - x0, y1 - y), z1 - z).view(-1, 1)
    wc = torch.mul(torch.mul(x1 - x, y - y0), z1 - z).view(-1, 1)
    wd = torch.mul(torch.mul(x - x0, y - y0), z1 - z).view(-1, 1)
    we = torch.mul(torch.mul(x1 - x, y1 - y), z - z0).view(-1, 1)
    wf = torch.mul(torch.mul(x - x0, y1 - y), z - z0).view(-1, 1)
    wg = torch.mul(torch.mul(x1 - x, y - y0), z - z0).view(-1, 1)
    wh = torch.mul(torch.mul(x - x0, y - y0), z - z0).view(-1, 1)

    interpolated_vol = torch.mul(wa, Ia) + torch.mul(wb, Ib) + torch.mul(wc, Ic) + torch.mul(wd, Id) +\
                       torch.mul(we, Ie) + torch.mul(wf, If) + torch.mul(wg, Ig) + torch.mul(wh, Ih)
    interpolated_vol[ind] = 0.0
    interpolated_vol = interpolated_vol.view(num_batch, out_depth, out_height, out_width, channels)
    interpolated_vol = interpolated_vol.permute(0, 4, 1, 2, 3)

    return interpolated_vol


def cal_ncc(I, J, eps):
    # compute local sums via convolution
    cross = (I - torch.mean(I)) * (J - torch.mean(J))
    I_var = (I - torch.mean(I)) * (I - torch.mean(I))
    J_var = (J - torch.mean(J)) * (J - torch.mean(J))

    cc = torch.sum(cross) / torch.sum(torch.sqrt(I_var*J_var + eps))

    test = torch.mean(cc)
    return torch.mean(cc)

# Gradient-NCC Loss
def gradncc(I, J, device='cuda', win=None, eps=1e-10):
        # compute filters
        with torch.no_grad():
            kernel_X = torch.Tensor([[[[1 ,0, -1],[2, 0 ,-2], [1, 0 ,-1]]]])
            kernel_X = torch.nn.Parameter( kernel_X, requires_grad = False )
            kernel_Y = torch.Tensor([[[[1, 2, 1],[0, 0, 0], [-1, -2 ,-1]]]])
            kernel_Y = torch.nn.Parameter( kernel_Y, requires_grad = False )
            SobelX = nn.Conv2d( 1, 1, 3, 1, 1, bias=False)
            SobelX.weight = kernel_X
            SobelY = nn.Conv2d( 1, 1, 3, 1, 1, bias=False)
            SobelY.weight = kernel_Y

            SobelX = SobelX.to(device)
            SobelY = SobelY.to(device)

        Ix = SobelX(I)
        Iy = SobelY(I)
        Jx = SobelX(J)
        Jy = SobelY(J)

        return  1-0.5*cal_ncc(Ix, Jx, eps)-0.5*cal_ncc(Iy, Jy, eps)

# NCC loss
def ncc(I, J, device='cuda', win=None, eps=1e-10):
    return 1-cal_ncc(I, J, eps)
