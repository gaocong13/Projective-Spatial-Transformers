import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchgeometry as tgm
import nibabel as nib
import cv2
from scipy.spatial.transform import Rotation
from scipy.optimize import curve_fit
import ProSTGrid

from posevec2mat import euler2mat

PI = 3.1415926
criterion = nn.MSELoss()

def hounsfield2linearatten(vol):
    vol = vol.astype(float)
    mu_water_ = 0.02683*1.0
    mu_air_   = 0.02485*0.0001
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
def input_param(CT_PATH, SEG_PATH, BATCH_SIZE, ISFlip = False, zRot90 = False, pix_spacing = 2.92, step_size = 1.75, iso_center = 400, norm_ct = False, device='cuda'):
    CT_vol_nib = nib.load(CT_PATH)
    _3D_vol_nib = nib.load(SEG_PATH)
    CT_vol = np.asanyarray(CT_vol_nib.dataobj)
    _3D_vol = np.asanyarray(_3D_vol_nib.dataobj)

    vol_affine = CT_vol_nib.affine
    # Currently assume volume spacing is isotropic
    # assert(abs(vol_affine[0][0]) == abs(vol_affine[1][1]) == abs(vol_affine[2][2]))
    vol_spacing = abs(vol_affine[0][0])

    # Rotation 90 degrees for making an AP view projection
    if zRot90:
        CT_vol = np.rot90(CT_vol, 3)
        _3D_vol = np.rot90(_3D_vol, 3)

    CT_vol = conv_hu_to_density(CT_vol)
    _3D_vol = CT_vol * (_3D_vol>0)

    if ISFlip:
        CT_vol = np.flip(CT_vol, axis=2)
        _3D_vol = np.flip(_3D_vol, axis=2)

    # Normalize CT
    if norm_ct:
        _3D_vol = (_3D_vol - np.min(_3D_vol)) / (np.max(_3D_vol) - np.min(_3D_vol))

    # Pre-defined hard coded geometry
    src_det = 1020
    det_size = 128
    # vol_size = CT_vol.shape[0]
    depth, height, width = CT_vol.shape

    norm_factor = (depth * vol_spacing / 2)
    src = (src_det - iso_center) / norm_factor
    det = -iso_center / norm_factor
    pix_spacing = pix_spacing / norm_factor
    step_size = step_size / norm_factor

    param = [src, det, pix_spacing, step_size, det_size]

    CT_vol = tensor_exp2torch(CT_vol, BATCH_SIZE, device)
    _3D_vol = tensor_exp2torch(_3D_vol, BATCH_SIZE, device)
    corner_pt = create_cornerpt(BATCH_SIZE, depth, height, width, device)
    ray_proj_mov = np.zeros((det_size, det_size))
    ray_proj_mov = tensor_exp2torch(ray_proj_mov, BATCH_SIZE, device)

    return param, det_size, _3D_vol, CT_vol, ray_proj_mov, corner_pt, norm_factor

def norm_target(BATCH_SIZE, det_size, target):
    min_tar, _ = torch.min(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
    max_tar, _ = torch.max(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
    target = (target.reshape(BATCH_SIZE, -1) - min_tar) / (max_tar - min_tar)
    target = target.reshape(BATCH_SIZE, 1, det_size, det_size)

    return target

def generate_fixed_grid(BATCH_SIZE, param, ray_proj_mov, corner_pt, norm_factor, device):
    src = param[0]
    det = param[1]
    pix_spacing = param[2]
    step_size = param[3]

    dist_min = 2.5
    bottom_cut = 0.5
    dist_max = src - det - bottom_cut
    grid = ProSTGrid.forward(corner_pt, ray_proj_mov.size(), dist_min, dist_max, src, det, pix_spacing, step_size, False)

    return grid

def init_rtvec_train(BATCH_SIZE, device):
     rtvec_gt = np.random.normal(0, 0.1, (BATCH_SIZE, 6))
     rtvec_gt[:, :3] = rtvec_gt[:, :3] * 0.35 * PI

     rtvec_smp = np.random.normal(0, 0.15, (BATCH_SIZE, 6))
     rtvec_smp[:, :3] = rtvec_smp[:, :3] * 0.35 * PI

     rtvec = rtvec_smp + rtvec_gt

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
     rtvec = rtvec.clone().detach().requires_grad_(True)

     rot_mat_gt = euler2mat(rtvec_gt_torch[:, :3])
     angle_axis_gt = tgm.rotation_matrix_to_angle_axis(torch.cat([rot_mat_gt,  torch.zeros(BATCH_SIZE, 3, 1).to(device)], dim=-1))
     rtvec_gt = torch.cat([angle_axis_gt, rtvec_gt_torch[:, 3:]], dim=-1)
     transform_mat4x4_gt = tgm.rtvec_to_pose(rtvec_gt)
     transform_mat3x4_gt = transform_mat4x4_gt[:, :3, :]

     return transform_mat3x4_gt, rtvec, rtvec_gt

def convert_numpy_euler_rtvec_to_ang_rtvec(euler_rtvec_np, device, req_grad=False):
    '''
    Args:
        euler_rtvec_np: rotation in euler radian (XYZ), translation in normalized geometry (XYZ)
        device: cpu or cuda
        req_grad: requires_grad or not

    Returns:
        ang_rtvec_torch: rtvec rotation in angle axis loaded to torch device
    '''
    BATCH_SIZE = euler_rtvec_np.shape[0]

    rtvec_torch = torch.tensor(euler_rtvec_np, dtype=torch.float, requires_grad=req_grad, device=device)
    rot_mat = euler2mat(rtvec_torch[:, :3])
    angle_axis = tgm.rotation_matrix_to_angle_axis(torch.cat([rot_mat,  torch.zeros(BATCH_SIZE, 3, 1).to(device)], dim=-1))
    ang_rtvec_torch = torch.cat([angle_axis, rtvec_torch[:, 3:]], dim=-1)

    return ang_rtvec_torch

def convert_numpy_euler_rtvec_to_mat3x4(euler_rtvec_np, device, req_grad=False):
    '''
    Args:
        euler_rtvec_np: rotation in euler radian (XYZ), translation in normalized geometry (XYZ)
        device: cpu or cuda
        req_grad: requires_grad or not

    Returns:
        transform_mat3x4: transformation matrix loaded in device
    '''
    BATCH_SIZE = euler_rtvec_np.shape[0]

    rtvec_torch = torch.tensor(euler_rtvec_np, dtype=torch.float, requires_grad=req_grad, device=device)
    rot_mat = euler2mat(rtvec_torch[:, :3])
    angle_axis = tgm.rotation_matrix_to_angle_axis(torch.cat([rot_mat,  torch.zeros(BATCH_SIZE, 3, 1).to(device)], dim=-1))
    ang_rtvec_torch = torch.cat([angle_axis, rtvec_torch[:, 3:]], dim=-1)

    transform_mat4x4 = tgm.rtvec_to_pose(ang_rtvec_torch)
    transform_mat3x4 = transform_mat4x4[:, :3, :]

    return transform_mat3x4

def convert_numpy_euler_rtvec_to_mat4x4(euler_rtvec_np, device, req_grad=False):
    '''
    Args:
        euler_rtvec_np: rotation in euler radian (XYZ), translation in normalized geometry (XYZ)
        device: cpu or cuda
        req_grad: requires_grad or not

    Returns:
        transform_mat4x4: transformation matrix loaded in device
    '''
    BATCH_SIZE = euler_rtvec_np.shape[0]

    rtvec_torch = torch.tensor(euler_rtvec_np, dtype=torch.float, requires_grad=req_grad, device=device)
    rot_mat = euler2mat(rtvec_torch[:, :3])
    angle_axis = tgm.rotation_matrix_to_angle_axis(torch.cat([rot_mat,  torch.zeros(BATCH_SIZE, 3, 1).to(device)], dim=-1))
    ang_rtvec_torch = torch.cat([angle_axis, rtvec_torch[:, 3:]], dim=-1)

    transform_mat4x4 = tgm.rtvec_to_pose(ang_rtvec_torch)

    return transform_mat4x4

def convert_numpy_euler_rtvec_to_center_mat4x4(euler_rtvec_np, device, req_grad=False):
    '''
    Args:
        euler_rtvec_np: rotation in euler radian (XYZ), translation in normalized geometry (XYZ)
        device: cpu or cuda
        req_grad: requires_grad or not

    Note:
        translational components xformed in original XYZ direction.

    Returns:
        transform_mat4x4: transformation matrix loaded in device
    '''
    BATCH_SIZE = euler_rtvec_np.shape[0]

    rtvec_torch = torch.tensor(euler_rtvec_np, dtype=torch.float, requires_grad=req_grad, device=device)
    rot_mat = euler2mat(rtvec_torch[:, :3])
    rot_mat4x4 = torch.eye(4, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1)
    rot_mat4x4[:, :3, :3] = rot_mat
    trans_mat4x4 = torch.eye(4, device=device).unsqueeze(0).repeat(BATCH_SIZE, 1, 1)
    trans_mat4x4[:, 0, -1] = rtvec_torch[:, 3]
    trans_mat4x4[:, 1, -1] = rtvec_torch[:, 4]
    trans_mat4x4[:, 2, -1] = rtvec_torch[:, 5]

    smp_mat4x4 = torch.matmul(rot_mat4x4, trans_mat4x4)
    tx = smp_mat4x4[:, 0, -1].view(-1, 1)
    ty = smp_mat4x4[:, 1, -1].view(-1, 1)
    tz = smp_mat4x4[:, 2, -1].view(-1, 1)

    angle_axis = tgm.rotation_matrix_to_angle_axis(smp_mat4x4[:, :3, :])
    ang_rtvec_torch = torch.cat([angle_axis, tx, ty, tz], dim=-1)

    transform_mat4x4 = tgm.rtvec_to_pose(ang_rtvec_torch)

    return transform_mat4x4

def convert_rtvec_to_transform_mat3x4_tgm(BATCH_SIZE, device, rtvec):
     rot_mat = euler2mat(rtvec[:, :3])
     angle_axis = tgm.rotation_matrix_to_angle_axis(torch.cat([rot_mat,  torch.zeros(BATCH_SIZE, 3, 1).to(device)], dim=-1))
     rtvec_cat = torch.cat([angle_axis, rtvec[:, 3:]], dim=-1)
     transform_mat4x4 = tgm.rtvec_to_pose(rtvec_cat)
     transform_mat3x4 = transform_mat4x4[:, :3, :]

     return transform_mat3x4

def convert_rtvec_to_transform_mat3x4(device, rtvec):
    BATCH_SIZE=1
    rtvec_torch = torch.tensor(rtvec, dtype=torch.float, requires_grad=True, device=device)
    rot = Rotation.from_rotvec(rtvec[:, :3])
    rot_mat = rot.as_matrix()

    rot_mat = torch.tensor(rot_mat, dtype=torch.float, requires_grad=True, device=device)
    angle_axis = tgm.rotation_matrix_to_angle_axis(torch.cat([rot_mat,  torch.zeros(BATCH_SIZE, 3, 1).to(device)], dim=-1))
    rtvec_torch = torch.cat([angle_axis, rtvec_torch[:, 3:]], dim=-1)
    transform_mat4x4 = tgm.rtvec_to_pose(rtvec_torch)
    transform_mat3x4 = transform_mat4x4[:, :3, :]

    return transform_mat3x4

def convert_transform_mat4x4_to_rtvec(transform_mat4x4):
     angle_axis = tgm.rotation_matrix_to_angle_axis(transform_mat4x4[:, :3, :])
     rtvec = torch.cat([angle_axis, transform_mat4x4[:, :3, -1]], dim=-1)

     return rtvec

def create_cornerpt(BATCH_SIZE, depth, height, width, device):
    sw = width / depth
    sh = height / depth
    sd = 1
    corner_pt = np.array([[-sw, -sh, -sd], [-sw, -sh, sd],
                          [-sw, sh, -sd],  [-sw, sh, sd],
                          [sw, -sh, -sd],  [sw, -sh, sd],
                          [sw, sh, -sd],   [sw, sh, sd]])
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
    scale_width = width / depth
    scale_height = height / depth
    scale_depth = 1
    # scale back the normalized grid coordinate (vol depth in [-1, 1]) to volume voxel coordinate
    x = width * (grid[:, :, :, :, 0] * 0.5 / scale_width + 0.5)
    y = height * (grid[:, :, :, :, 1] * 0.5 / scale_height + 0.5)
    z = depth * (grid[:, :, :, :, 2] * 0.5 / scale_depth + 0.5)

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

def fit_gaussian(image):
    plot_fig = False
    if plot_fig:
        plt.figure()
        plt.imshow(image,cmap="gray")  # x-ray image from DeepDRR
        plt.title("Original Image")
        plt.show()

    num_bins = 2000
    hx, hy = np.histogram(image.flatten(), bins=num_bins, density=True)

    # popt, pcov = curve_fit(Gauss, hy[0:256], hx, p0 = [30000, 2, 1], maxfev=5000)  # Gaussian fit with initial guess
    caty = np.concatenate((-np.flip(hy[0:1000]), hy[1::]), axis=0)
    catx = np.concatenate((np.zeros(1000), hx), axis=0)
    popt, pcov = curve_fit(Gauss, caty, catx, maxfev=5000)
    x0 = popt[1]  # mean of fitted Gaussian
    sigma = popt[2]  # sigma of fitted Gaussian
    relative_mean = x0/np.max(hy)
    relative_sigma = sigma/np.max(hy)
    print("relative mean:", relative_mean)
    print("relative sigma:", relative_sigma)
    sigma_mean_ratio = abs(sigma/x0)
    print("ratio:", sigma_mean_ratio)

    dx = hy[1] - hy[0]
    AccuHist = np.cumsum(hx)*dx

    if abs(relative_sigma) < 0.005:
        threshold_accuhist = 0.85 + 0.1 * abs(relative_sigma) / 0.005
    elif abs(relative_sigma) < 0.08:
        threshold_accuhist = 0.9 + 0.08 * abs(relative_sigma) / 0.08
    else:
        threshold_accuhist = 0.98

    minarg_accuhist = np.argwhere(AccuHist > threshold_accuhist)
    cutoff_intensity = hy[minarg_accuhist[0]]
    #Dec.24: cutoff_intensity = x0 + 8 * 0.4 * np.abs(sigma) / sigma_mean_ratio
    '''
    if abs(relative_sigma) > 0.01 or relative_mean > 0.05:
        cutoff_intensity = x0 + 5 * np.abs(sigma) / sigma_mean_ratio  # Select the cutoff intensity at x0 + 3sigma,
        #                                            # which covers 99.74% of Gaussian Distribution
        # cutoff_intensity = 0.9 * np.max(hy)
    else:
        cutoff_intensity = 0.2 * np.max(hy)
    '''
    print('Cutoff_intensity from Gaussian: ', cutoff_intensity)

    #### Display of histogram with Gaussian fit cut
    if True:
        plt.figure()
        plt.plot(caty, catx, 'g+:', label='original histogram')

        plt.plot(hy[1:], AccuHist, 'c+:', label='cumulative histogram')
        #plt.plot(hy, Gauss(hy, *popt), 'r-', label='Gaussian fit')
        fit_vals = Gauss(hy[1:], *popt)
        plt.plot(hy[1:],  fit_vals, 'r-', label="Gaussian fit: mean: {:.4f} + sigma: {:.4f}".format(relative_mean, relative_sigma))
        plt.plot(cutoff_intensity, 0, markersize=8, marker='o', color='r', label="Cutoff point: {:.4f} Accuhist threshold: {:.4f}".format(cutoff_intensity[0], threshold_accuhist))
        plt.title('Histogram')
        plt.legend(loc='upper right')
        plt.show()
        # plt.savefig('/home/cong/Research/Generalization/H5_File/NewMexico_20CT/gauss_thred/' + pose + '.png')
        # plt.close()

    return cutoff_intensity

# Define a Gaussian Distribution
def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

# Define a Gaussian Distribution
def Exp(x, a):
    return a * np.exp(-a * x)
