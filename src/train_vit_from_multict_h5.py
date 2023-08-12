from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchgeometry as tgm
import numpy as np
from numpy import linalg as LA
import shutil

import logging
from rich.logging import RichHandler
import sys
import os
import argparse
import h5py
import copy
from torch.autograd import Variable
from torch.optim.lr_scheduler import CyclicLR
from scipy import spatial

from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})
import random

from module import ProST_init_multict, RegiNet_multict, RegiNet3d_multict
from module_grid import ProST_init_multict_grid, RegiNet_multict_grid

from module_vit import RegiNet_CrossViTv2_SW
from posevec2mat import euler2mat
from util import gradncc, count_parameters, norm_target, generate_fixed_grid, convert_numpy_euler_rtvec_to_mat4x4, \
                 convert_transform_mat4x4_to_rtvec
from util_plot import plot_run_stat_mse, plot_run_stat_geo, plot_geomag_run_stat
from util_aug import aug_torch_target

from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric
import geomstats.geometry.riemannian_metric as riem
from loss_geodesic import Geodesic_loss

device = torch.device("cuda")
PI = 3.1415926
DEG2RAD = PI/180.0
RAD2DEG = 180.0/PI
NUM_PHOTON = 20000
EPS = 1e-10

clipping_value = 10

SE3_GROUP = SpecialEuclidean(n=3, point_type='vector')
RiemMetric = RiemannianMetric(dim=6)
METRIC = SE3_GROUP.left_canonical_metric
riem_dist_fun = RiemMetric.dist

zFlip = False
MICCAIgeom = True
valid_offset_rot = 50.
valid_offset_trans = 60.
valid_transZoffset_scale = 5 # valid_offset_transZ = valid_offset_trans * valid_transZoffset_scale
valid_num_sample = 50
VALID_BATCH_SIZE = 1
valid_iter_plot = False
# Every 10 epochs clear loss cache list
EPOCH_LOSS_CACHE = 10

# Fix random seed:
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(seed)
np.random.seed(seed)

def rand_smp_rtvec(euler_rtvec_gt):
    smp_sigma = 0.15
    euler_rtvec_smp = np.random.normal(0, smp_sigma, (BATCH_SIZE, 6))
    # euler_rtvec_smp = np.clip(euler_rtvec_smp, -2*smp_sigma, 2*smp_sigma)
    euler_rtvec_smp[:, :3] = euler_rtvec_smp[:, :3] * 0.35 * PI

    euler_rtvec_init = euler_rtvec_smp + euler_rtvec_gt
    rtvec_torch = torch.tensor(euler_rtvec_init, dtype=torch.float, requires_grad=True, device=device)
    rtvec_gt_torch = torch.tensor(euler_rtvec_gt, dtype=torch.float, requires_grad=True, device=device)

    rot_mat = euler2mat(rtvec_torch[:, :3])
    angle_axis = tgm.rotation_matrix_to_angle_axis(torch.cat([rot_mat,  torch.zeros(BATCH_SIZE, 3, 1).to(device)], dim=-1))
    rtvec = torch.cat([angle_axis, rtvec_torch[:, 3:]], dim=-1)

    rot_mat_gt = euler2mat(rtvec_gt_torch[:, :3])
    angle_axis_gt = tgm.rotation_matrix_to_angle_axis(torch.cat([rot_mat_gt,  torch.zeros(BATCH_SIZE, 3, 1).to(device)], dim=-1))
    rtvec_gt = torch.cat([angle_axis_gt, rtvec_gt_torch[:, 3:]], dim=-1)

    return rtvec, rtvec_gt

def rand_smp_ang_rtvec(ang_rtvec_gt, norm_factor):
    euler_rtvec_smp = np.concatenate((np.random.normal(0, 25, (BATCH_SIZE, 3)), # Sample euler rotation
                                        np.random.normal(0, 25, (BATCH_SIZE, 2)), # Sample in-plane translation
                                        np.random.normal(0, 60, (BATCH_SIZE, 1))), # Sample depth translation
                                        axis=-1)
    euler_rtvec_smp[:, :3] = euler_rtvec_smp[:, :3] * DEG2RAD
    euler_rtvec_smp[:, 3:] = euler_rtvec_smp[:, 3:] / norm_factor
    with torch.no_grad():
        init_mat4x4 = convert_numpy_euler_rtvec_to_mat4x4(euler_rtvec_smp, device)
        rtvec_init = convert_transform_mat4x4_to_rtvec(init_mat4x4)

    rtvec_init.requires_grad = True
    rtvec_gt = torch.tensor(ang_rtvec_gt, dtype=torch.float, requires_grad=True, device=device)

    return rtvec_init, rtvec_gt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ProST Training Scripts. Settings used for training and log setup',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('save_path', help='Root folder to save training results', type=str)
    parser.add_argument('h5_file', help='H5 file that contains CT and image data', type=str)
    parser.add_argument('--resume-epoch', help='Resumed epoch used for continuing training. -1 means starting from scratch', type=int, default=-1)
    parser.add_argument('--batch-size', help='Batch size', type=int, default=2)
    parser.add_argument('--step-size', help='Step size', type=float, default=2.0)
    parser.add_argument('--iter-num', help='Number of iterations per epoch', type=int, default=50)
    parser.add_argument('--epoch-num', help='Number of epochs for training', type=int, default=2000)
    parser.add_argument('--save-freq', help='Number of epochs to save a model to disk', type=int, default=100)
    parser.add_argument('--loss-info-freq', help='Number of epochs to plot loss fig and save to disk', type=int, default=50)
    parser.add_argument('--normct', help='Normalize CT intensities', action='store_true', default=True)
    parser.add_argument('--normmov', help='Normalize ProST moving image', action='store_true', default=False)
    parser.add_argument('--no-3d-ori', help='Not performing ResNet 3D CT connection', action='store_true', default=False)
    parser.add_argument('--no-3d-net', help='No 3D network for ablation study', action='store_true', default=False)
    parser.add_argument('--geo-mag-loss', help='Apply geodesic magnitude as regression loss', action='store_true', default=False)
    parser.add_argument('--grad-mag-loss', help='Use gradient magnitude not only direction as loss', action='store_true', default=False)
    parser.add_argument('--grad-geo-loss', help='Compute loss using geodesic loss between two gradient vectors', action='store_true', default=False)
    parser.add_argument('--aug', help='Perform image augmentation', action='store_true', default=False)
    parser.add_argument('--iter-fig-freq', help='Number of iterations to save a debug plot figure to disk folder', type=int, default=50)
    parser.add_argument('--debug-plot', help='Plotting iterative moving image during sampling', action='store_true', default=False)
    parser.add_argument('-n', '--no-writing-to-disk', help='Not creating folders or writing anything to disk', action='store_true', default=False)
    parser.add_argument('--log-nan-tensor', help='Log related tensors to disk if nan happens', action='store_true', default=False)
    parser.add_argument('--tensorboard_writer', help='Enable tensorboard writer', action='store_true', default=False)
    parser.add_argument('--valid', help='Perform validation to plot loss shape', action='store_true', default=False)
    parser.add_argument('--val-freq', help='Frequency to perform sim loss validation', type=int, default=100)
    parser.add_argument('--val-ct-id', help='Validation CT Index in the CT List', type=int, default=0)
    parser.add_argument('--val-proj-id', help='Validation Projection Index in the CT group', type=int, default=0)
    parser.add_argument('--fixed-grid', help='Use a fixed generated grid in avoid of MARCC memory issue', action='store_true', default=False)
    parser.add_argument('--ang-normal-smp', help='Normal sampling moving data pose on angle axis', action='store_true', default=False)
    parser.add_argument('--no-sim-norm', help='Do not normalize similarity metric during validation loss plot', action='store_true', default=True)
    parser.add_argument('--same-batch-target', help='Same target image within a batch', action='store_true', default=False)
    parser.add_argument('--cuda-id', help='Specify CUDA id when training on pong', type=str, default="")

    args = parser.parse_args()

    SAVE_PATH = args.save_path
    H5_File = args.h5_file
    net = 'CrossViTv2_SW'
    RESUME_EPOCH = args.resume_epoch
    BATCH_SIZE = args.batch_size
    STEP_SIZE = args.step_size
    ITER_NUM = args.iter_num
    END_EPOCH = args.epoch_num
    SAVE_MODEL_EVERY_EPOCH = args.save_freq
    LOSS_INFO_EVERY_EPOCH = args.loss_info_freq
    SAVE_DEBUG_PLOT_EVERY_ITER = args.iter_fig_freq
    NORM_CT = args.normct
    NORM_MOV = args.normmov
    NO_3D_ORI = args.no_3d_ori
    NO_3D_NET = args.no_3d_net
    geo_mag_loss = args.geo_mag_loss
    grad_mag_loss = args.grad_mag_loss
    grad_geo_loss = args.grad_geo_loss
    do_aug = args.aug
    debug_plot = args.debug_plot
    writing_to_disk = not args.no_writing_to_disk
    log_nan_tensor = args.log_nan_tensor
    log_tensorboard_writer = args.tensorboard_writer
    valid_sim_loss = args.valid
    VALID_SIM_LOSS_EPOCH = args.val_freq
    val_ct_id = args.val_ct_id
    val_proj_id = args.val_proj_id
    use_fixed_grid = args.fixed_grid
    no_valid_sim_norm = args.no_sim_norm
    same_batch_target = args.same_batch_target
    cuda_id = args.cuda_id

    # This is specifically to run on pong
    if cuda_id:
        os.environ["CUDA_VISIBLE_DEVICES"]=cuda_id

    checkpoint_folder = SAVE_PATH + '/checkpoint'
    resumed_checkpoint_filename = checkpoint_folder + '/vali_model'+str(RESUME_EPOCH)+'.pt'

    tensorboard_writer_folder = SAVE_PATH + "/run_log_dir"
    log_nan_tensor_file = SAVE_PATH + "/run_log_dir/saved_tensor"
    stat_figs_folder = SAVE_PATH + "/stat_figs"
    sim_loss_debug_folder = SAVE_PATH + "/sim_loss_figs"
    iter_proj_debug_folder = SAVE_PATH + "/iter_projs"

    log = logging.getLogger('deepdrr')
    # log.setLevel(logging.DEBUG)
    log.propagate = False

    if RESUME_EPOCH>=0:
        prev_state = torch.load(resumed_checkpoint_filename)

        print('loading training params from checkpoint state dict...')
        # SAVE_PATH                   = prev_state['save-path']
        # H5_File                     = prev_state['h5-file']
        STEP_SIZE                   = prev_state['grid-step-size']
        ITER_NUM                    = prev_state['iter-num']
        START_EPOCH                 = prev_state['epoch'] + 1
        step_cnt                    = prev_state['epoch'] * ITER_NUM
        # END_EPOCH                   = prev_state['end-epoch']
        SAVE_MODEL_EVERY_EPOCH      = prev_state['save-freq']
        SAVE_DEBUG_PLOT_EVERY_ITER  = prev_state['debug-plot-freq']
        VALID_SIM_LOSS_EPOCH        = prev_state['val-freq']
        NORM_CT                     = prev_state['norm-ct']
        NORM_MOV                    = prev_state['norm-mov']
        NO_3D_ORI                   = prev_state['no-3d-ori']
        NO_3D_NET                   = prev_state['no-3d-net']
        debug_plot                  = prev_state['debug-plot']
        writing_to_disk             = prev_state['writing-to-disk']
        log_nan_tensor              = prev_state['log-nan-tensor']
        log_tensorboard_writer      = prev_state['tensorboard-writer']
        # valid_sim_loss              = prev_state['valid-sim-loss']
        val_ct_id                   = prev_state['valid-ct-id']
        val_proj_id                 = prev_state['valid-proj-id']
        use_fixed_grid              = prev_state['use-fixed-grid']
        BATCH_SIZE                  = prev_state['batch-size']
        do_aug                      = prev_state['data-aug']
        geo_mag_loss                = prev_state['geo-mag-loss']
        grad_mag_loss               = prev_state['grad-mag-loss']
        grad_geo_loss               = prev_state['grad-geo-loss']
        np.random.set_state(prev_state['numpy-random-state'])
        torch.set_rng_state(prev_state['torch-random-state'])
        torch.cuda.set_rng_state(prev_state['cuda-random-state'])
    else:
        START_EPOCH = 0
        step_cnt = 0

    if writing_to_disk:
        if not os.path.exists(SAVE_PATH):
            print('Creating...' + SAVE_PATH)
            os.mkdir(SAVE_PATH)

        if not os.path.exists(checkpoint_folder):
            print('Creating...' + checkpoint_folder)
            os.mkdir(checkpoint_folder)

        if not os.path.exists(tensorboard_writer_folder):
            print('Creating...' + tensorboard_writer_folder)
            os.mkdir(tensorboard_writer_folder)

        if not os.path.exists(log_nan_tensor_file):
            print('Creating...' + log_nan_tensor_file)
            os.mkdir(log_nan_tensor_file)

        if not os.path.exists(stat_figs_folder):
            print('Creating...' + stat_figs_folder)
            os.mkdir(stat_figs_folder)

        if not os.path.exists(sim_loss_debug_folder):
            print('Creating...' + sim_loss_debug_folder)
            os.mkdir(sim_loss_debug_folder)

        if debug_plot and not os.path.exists(iter_proj_debug_folder):
            print('Creating...' + iter_proj_debug_folder)
            os.mkdir(iter_proj_debug_folder)

        with open(SAVE_PATH + '/printlog.txt', 'a') as f:
            f.write('\nNew restart...\n')

        with open(SAVE_PATH + '/printout.txt', 'a') as f:
            f.write('\nNew restart...\n')

    hf = h5py.File(H5_File, 'r')

    CT_DATA_LIST = hf.get("ct-list")
    num_ct = len(CT_DATA_LIST)
    print("H5 file loaded:\n", H5_File)
    print("Number of CTs:\n", num_ct)

    # Training Setup
    criterion_mse = nn.MSELoss()

    encoder_share_weights = False
    if not use_fixed_grid:
        if net == "miccai":
            model = RegiNet_multict(log_nan_tensor_file, debug_plot, NORM_MOV, no_3d_net=NO_3D_NET).to(device)
        elif net == "miccai3d":
            model = RegiNet3d_multict(log_nan_tensor_file, debug_plot, NORM_MOV).to(device)
        elif net == "CrossViTv2_SW":
            model = RegiNet_CrossViTv2_SW(log_nan_tensor_file, debug_plot, NORM_MOV, NO_3D_ORI, NO_3D_NET).to(device)
            encoder_share_weights = True
        else:
            sys.exit("Network not implemented!")
    else:
        if net == "miccai":
            model = RegiNet_multict_grid(log_nan_tensor_file, debug_plot, NORM_MOV).to(device)
        else:
            sys.exit("Network not implemented!")

    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    scheduler = CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-4,step_size_up=100)

    if RESUME_EPOCH >= 0:
        print('Resuming model from', resumed_checkpoint_filename)
        model.load_state_dict(prev_state['model-state-dict'])
        optimizer.load_state_dict(prev_state['optimizer-state-dict'])
        scheduler.load_state_dict(prev_state['scheduler-state-dict'])

    print('module parameters:', count_parameters(model))

    if log_tensorboard_writer:
        writer = SummaryWriter(tensorboard_writer_folder)

    # Start -- Generate validation target image
    if valid_sim_loss:
        VAL_CT_NAME = CT_DATA_LIST[val_ct_id]
        ct_grp = hf[VAL_CT_NAME]

        num_proj        = ct_grp.get("num-proj")[()]
        param           = np.array(ct_grp.get("param"))
        det_size        = ct_grp.get("det-size")[()]
        _3D_vol_np      = np.array(ct_grp.get("ctseg-vol"))
        CT_vol_np       = np.array(ct_grp.get("ct-vol"))
        ray_proj_mov_np = np.array(ct_grp.get("ray-proj-mov"))
        corner_pt_np    = np.array(ct_grp.get("corner-pt"))
        norm_factor     = ct_grp.get("norm-factor")[()]

        criterion_gradncc = gradncc

        val_param        = param
        val_det_size     = det_size
        val_3D_vol       = torch.tensor(np.repeat(_3D_vol_np, 1, axis=0), dtype=torch.float, requires_grad=True, device=device)
        val_CT_vol       = torch.tensor(np.repeat(CT_vol_np, 1, axis=0), dtype=torch.float, requires_grad=True, device=device)
        val_ray_proj_mov = torch.tensor(np.repeat(ray_proj_mov_np, 1, axis=0), dtype=torch.float, requires_grad=True, device=device)
        val_corner_pt    = torch.tensor(np.repeat(corner_pt_np, 1, axis=0), dtype=torch.float, requires_grad=True, device=device)
        val_norm_factor  = norm_factor

        initmodel = ProST_init_multict(log_nan_tensor_file).to(device) if not use_fixed_grid else ProST_init_multict_grid(log_nan_tensor_file).to(device)

        if use_fixed_grid:
            val_grid = generate_fixed_grid(1, val_param, val_ray_proj_mov, val_corner_pt, val_norm_factor, device)

        val_proj_name = str(val_proj_id).zfill(5)
        val_proj_grp = ct_grp[val_proj_name]
        target_valid_arr = np.zeros((1, 128, 128))
        target_valid_arr[0] = np.array(val_proj_grp.get("proj"))
        target_valid = torch.tensor(target_valid_arr[0], dtype=torch.float, requires_grad=True, device=device)
        target_valid = norm_target(1, val_det_size, target_valid)

        rtvec_val_gt_arr = np.zeros((1, 6))
        rtvec_val_gt_arr[0] = np.array(val_proj_grp.get("angvec"))
        rtvec_val_gt = torch.tensor(rtvec_val_gt_arr, dtype=torch.float, device=device)
        transform_mat4x4_val_gt = tgm.rtvec_to_pose(rtvec_val_gt)

        val_gradncc_sim_list_allDOF = list()
        val_xdim_list_allDOF = list()

        print('Generating GradNCC Validation loss...')
        with torch.no_grad():
            for offsetDOF in range(6):
                    if valid_iter_plot:
                        fig_valid = plt.figure(figsize=(18,6))

                    if offsetDOF in [0, 1, 2]:
                        offsetmag = valid_offset_rot * DEG2RAD
                    elif offsetDOF in [3, 4]:
                        offsetmag = valid_offset_trans / val_norm_factor
                    elif offsetDOF in [5]:
                        offsetmag = valid_transZoffset_scale * valid_offset_trans / val_norm_factor
                    else:
                        sys.exit("offsetDOF is not in range!")

                    val_gradncc_sim_list = []

                    euler_rtvec_smp = np.zeros((1, 6))
                    for idz in range(valid_num_sample):
                        euler_rtvec_smp[0, offsetDOF] = -offsetmag + idz*(2*offsetmag)/valid_num_sample
                        smp_mat4x4 = convert_numpy_euler_rtvec_to_mat4x4(euler_rtvec_smp, device)
                        mat4x4 = torch.matmul(smp_mat4x4, transform_mat4x4_val_gt)

                        with torch.no_grad():
                            rtvec_val = convert_transform_mat4x4_to_rtvec(mat4x4)

                        # Do Projection
                        transform_mat4x4_valid = tgm.rtvec_to_pose(rtvec_val)
                        transform_mat3x4_valid = transform_mat4x4_valid[:, :3, :]
                        vals = initmodel(val_3D_vol, val_ray_proj_mov, transform_mat3x4_valid, val_corner_pt, val_param) if not use_fixed_grid \
                            else initmodel(val_3D_vol, val_ray_proj_mov, transform_mat3x4_valid, val_grid)
                        proj_mov = vals[0]

                        # gradncc Similarity:
                        gradncc_loss = criterion_gradncc(proj_mov, target_valid)

                        val_gradncc_sim_list.append(gradncc_loss.item())

                        # Plot Moving Image
                        if valid_iter_plot:
                            proj_mov_numpy0 = np.array(proj_mov[0,0,:,:].data.cpu())
                            target_numpy0 = np.array((target_valid[0,0,:,:]).data.cpu())
                            proj_mov_numpy0 = proj_mov_numpy0.reshape((val_det_size , val_det_size))
                            target_numpy0 = target_numpy0.reshape((val_det_size , val_det_size))

                            fig_valid.suptitle('Iter: ' + str(idz) + '/' + str(valid_num_sample) )
                            ax_mov = fig_valid.add_subplot(131)
                            ax_tar = fig_valid.add_subplot(132)
                            ax_loss = fig_valid.add_subplot(133)
                            ax_mov.imshow(proj_mov_numpy0)
                            ax_mov.set_title('Moving Image')

                            ax_tar.imshow(target_numpy0)
                            ax_tar.set_title('Target Image')

                            ax_loss.plot(val_gradncc_sim_list, 'ro-')
                            ax_loss.set_title('Loss')
                            ax_loss.legend(['GradNCC Similarity'])

                            plt.show(block=False)
                            plt.pause(0.2)
                            plt.clf()

                    if valid_iter_plot:
                        plt.close(fig_valid)

                    if offsetDOF in [0, 1, 2]:
                        val_xdim = np.linspace(-valid_offset_rot, valid_offset_rot, valid_num_sample)
                    elif offsetDOF in [3, 4]:
                        val_xdim = np.linspace(-valid_offset_trans, valid_offset_trans, valid_num_sample)
                    elif offsetDOF in [5]:
                        val_xdim = valid_transZoffset_scale * np.linspace(-valid_offset_trans, valid_offset_trans, valid_num_sample)

                    val_gradncc_sim_list_allDOF.append(val_gradncc_sim_list)
                    val_xdim_list_allDOF.append(val_xdim)

        # remove model from GPU to release memory
        del initmodel

        plots_xlabel_txt = {
            0: 'X Rotation (degrees)',
            1: 'Y Rotation (degrees)',
            2: 'Z Rotation (degrees)',
            3: 'X Translation (mm)',
            4: 'Y Translation (mm)',
            5: 'Z Translation (mm)',
        }

        plots_zlabel_txt = {
            0: 'Similarity Shape - Rotation X',
            1: 'Similarity Shape - Rotation Y',
            2: 'Similarity Shape - Rotation Z',
            3: 'Similarity Shape - Translation X',
            4: 'Similarity Shape - Translation Y',
            5: 'Similarity Shape - Translation Z',
        }

        torch.cuda.empty_cache()
    # End -- Generate validation target image


    riem_dist_list = []

    if not geo_mag_loss:
        mse_loss_list = []
        riem_dist_mean_list = []
        total_loss_list = []
        riem_grad_loss_list = []
        vecgrad_diff_list = []
        rtvec_grad_rot_dist_list = []
        rtvec_grad_trans_dist_list = []
        rtvec_grad_rot_norm_list = []
        rtvec_grad_trans_norm_list = []
        riem_grad_rot_norm_list = []
        riem_grad_trans_norm_list = []
        if grad_geo_loss:
            riem_grad_loss_all_list = []
        else:
            riem_grad_rot_loss_all_list = []
            riem_grad_trans_loss_all_list = []
    else:
        riem_dist_loss_list = []
        mse_loss_diff_list = []

    training_loss_ylim = 10

    print('Start training...')
    for epoch in range(START_EPOCH, END_EPOCH+1):
        ## Do Iterative Validation
        model.train()
        model.require_grad = True

        # Load CT from h5 and train for 1 epoch
        ct_id = random.randint(1, num_ct-1)
        CT_NAME = CT_DATA_LIST[ct_id]
        ct_grp = hf[CT_NAME]

        num_proj        = ct_grp.get("num-proj")[()]
        param           = np.array(ct_grp.get("param"))
        det_size        = ct_grp.get("det-size")[()]
        _3D_vol_np      = np.array(ct_grp.get("ctseg-vol"))
        CT_vol_np       = np.array(ct_grp.get("ct-vol"))
        ray_proj_mov_np = np.array(ct_grp.get("ray-proj-mov"))
        corner_pt_np    = np.array(ct_grp.get("corner-pt"))
        norm_factor     = ct_grp.get("norm-factor")[()]

        if NORM_CT:
            _3D_vol_np = (_3D_vol_np - np.min(_3D_vol_np))/(np.max(_3D_vol_np) - np.min(_3D_vol_np))

        _3D_vol = torch.tensor(np.repeat(_3D_vol_np, BATCH_SIZE, axis=0), dtype=torch.float, requires_grad=True, device=device)
        CT_vol  = torch.tensor(np.repeat(CT_vol_np, BATCH_SIZE, axis=0), dtype=torch.float, requires_grad=True, device=device)
        ray_proj_mov = torch.tensor(np.repeat(ray_proj_mov_np, BATCH_SIZE, axis=0), dtype=torch.float, requires_grad=True, device=device)
        corner_pt = torch.tensor(np.repeat(corner_pt_np, BATCH_SIZE, axis=0), dtype=torch.float, requires_grad=True, device=device)

        if use_fixed_grid:
            grid = generate_fixed_grid(BATCH_SIZE, param, ray_proj_mov, corner_pt, norm_factor, device)

        for iter in range(ITER_NUM):
            step_cnt = step_cnt+1

            # Load target image from h5 file
            rtvec_gt_arr = np.zeros((BATCH_SIZE, 6))
            euler_rtvec_smp_arr = np.zeros((BATCH_SIZE, 6))
            euler_rtvec_gt_arr = np.zeros((BATCH_SIZE, 6))
            target_arr = np.zeros((BATCH_SIZE, 128, 128))

            # Keep target image the same within a batch
            if same_batch_target:
                target_proj_ID = random.randint(0, num_proj-1)
                target_proj_name = str(target_proj_ID).zfill(5)
                proj_grp = ct_grp[target_proj_name]
            for b in range(BATCH_SIZE):
                if not same_batch_target:
                    target_proj_ID = random.randint(0, num_proj-1)
                    target_proj_name = str(target_proj_ID).zfill(5)
                    proj_grp = ct_grp[target_proj_name]

                rtvec_gt_arr[b, :] = np.array(proj_grp.get("angvec"))
                euler_rtvec_gt_arr[b, :] = np.array(proj_grp.get("eulervec"))
                target_arr[b, :, :] = np.array(proj_grp.get("proj"))
                # euler_rtvec_smp_arr[b, :] = rand_smp_rtvec(norm_factor)

            rtvec, rtvec_gt = rand_smp_ang_rtvec(rtvec_gt_arr, norm_factor)

            # Augment the target image
            # target_arr = aug_target(target_arr)
            with torch.no_grad():
                target = torch.tensor(target_arr, dtype=torch.float, requires_grad=True, device=device)
                target = norm_target(BATCH_SIZE, det_size, target)

            if do_aug:
                target = aug_torch_target(target, device)

            # Do Projection and get two encodings
            vals = model(_3D_vol, target, rtvec, corner_pt, param, log_nan_tensor=log_nan_tensor) if not use_fixed_grid \
                else model(_3D_vol, target, rtvec, grid, log_nan_tensor=log_nan_tensor)

            if type(vals) == bool and (not vals):# and (not vals[0])
                continue
            elif len(vals) == 3 and vals[-1] and (not encoder_share_weights):
                encode_mov = vals[0]
                encode_tar = vals[1]
            elif debug_plot and len(vals) == 4 and vals[-1] and (not encoder_share_weights):
                encode_mov = vals[0]
                encode_tar = vals[1]
                proj_mov = vals[2]
            elif len(vals) == 2 and vals[-1] and encoder_share_weights:
                encode_out = vals[0]
            elif debug_plot and len(vals) == 3 and vals[-1] and encoder_share_weights:
                encode_out = vals[0]
                proj_mov = vals[1]
            else:
                print('Invalid Model Return!!!!!')
                continue

            if debug_plot:
                debug_fig, debug_axes = plt.subplots(BATCH_SIZE, 2)
                debug_fig.suptitle("CT: " + str(CT_NAME) + " Proj:" + target_proj_name)
                if BATCH_SIZE > 1:
                    for ax_idx in range(BATCH_SIZE):
                        debug_axes[ax_idx, 0].imshow(proj_mov.cpu().detach().numpy()[ax_idx, 0, :, :])
                        debug_axes[ax_idx, 0].set_title("mov" + str(ax_idx))

                        debug_axes[ax_idx, 1].imshow(target.detach().cpu().numpy()[ax_idx, 0, :, :])
                        debug_axes[ax_idx, 1].set_title("tar" + str(ax_idx))
                else:
                    debug_axes[0].imshow(proj_mov.cpu().detach().numpy()[0, 0, :, :])
                    debug_axes[0].set_title("mov")

                    debug_axes[1].imshow(target.detach().cpu().numpy()[0, 0, :, :])
                    debug_axes[1].set_title("tar")

                # plt.hist(target.detach().cpu().numpy()[0, 0, :, :].flatten(), bins=50)
                # plt.show()
                if writing_to_disk and (iter % SAVE_DEBUG_PLOT_EVERY_ITER == 0):
                    plt.savefig(iter_proj_debug_folder + "/epoch" + str(epoch).zfill(4) + "_iter" + str(iter).zfill(3) + ".png" )

                plt.close('all')

            optimizer.zero_grad()

            # Find geodesic distance
            riem_dist = np.sqrt(riem.loss(rtvec.detach().cpu(), rtvec_gt.detach().cpu(), METRIC))

            if geo_mag_loss:
                # Calculate Net l2 Loss, L_N
                l2_loss = encode_out if encoder_share_weights else criterion_mse(encode_mov, encode_tar) #RegiNet loss

                riem_dist_torch = torch.tensor(riem_dist, dtype=torch.float, requires_grad=False, device=device)
                riem_dist_loss = criterion_mse(l2_loss.view(-1), riem_dist_torch.view(-1))
                riem_dist_loss.backward()
                # Clip training gradient magnitude
                torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
                optimizer.step()

                mse_loss_diff_list.append(torch.abs(l2_loss.view(-1) - riem_dist_torch.view(-1)).detach().cpu().numpy())
                riem_dist_list.append(riem_dist)
                riem_dist_loss_list.append(riem_dist_loss.detach().item())
            else:
                # Calculate Net l2 Loss, L_N
                l2_loss = torch.mean(encode_out) if encoder_share_weights else criterion_mse(encode_mov, encode_tar) #RegiNet loss

                z = Variable(torch.ones(l2_loss.shape)).to(device)

                rtvec_grad = torch.autograd.grad(l2_loss, rtvec, grad_outputs=z, only_inputs=True, create_graph=True,
                                                      retain_graph=True)[0]

                # Find geodesic gradient
                riem_grad = riem.grad(rtvec.detach().cpu(), rtvec_gt.detach().cpu(), METRIC)
                riem_grad = torch.tensor(riem_grad, dtype=torch.float, requires_grad=False, device=device)

                if grad_geo_loss:
                    riem_grad_loss = Geodesic_loss.apply(rtvec_grad, riem_grad)
                else:
                    if grad_mag_loss:
                        ### Translation Loss
                        riem_grad_trans_loss = torch.mean(torch.sum((riem_grad[:, 3:] - rtvec_grad[:, 3:])**2, dim=-1))
                        riem_grad_transZ_loss = torch.mean(torch.sum((riem_grad[:, -1] - rtvec_grad[:, -1])**2, dim=-1))

                        ### Rotation Loss
                        riem_grad_rot_loss = torch.mean(torch.sum((riem_grad[:, :3] - rtvec_grad[:, :3])**2, dim=-1))
                        riem_grad_rotZ_loss = torch.mean(torch.sum((riem_grad[:, 2] - rtvec_grad[:, 2])**2, dim=-1))
                    else:
                        ### Translation Loss
                        riem_grad_transnorm = riem_grad[:, 3:]/(torch.norm(riem_grad[:, 3:], dim=-1, keepdim=True)+EPS)
                        rtvec_grad_transnorm = rtvec_grad[:, 3:]/(torch.norm(rtvec_grad[:, 3:], dim=-1, keepdim=True)+EPS)
                        riem_grad_trans_loss = torch.mean(torch.sum((riem_grad_transnorm - rtvec_grad_transnorm)**2, dim=-1))
                        riem_grad_transZ_loss = torch.mean(torch.sum((riem_grad_transnorm[:, -1] - rtvec_grad_transnorm[:, -1])**2, dim=-1))
                        # riem_grad_trans_loss = -torch.mean(F.cosine_similarity(riem_grad_transnorm, rtvec_grad_transnorm, dim=-1))

                        ### Rotation Loss
                        riem_grad_rotnorm = riem_grad[:, :3]/(torch.norm(riem_grad[:, :3], dim=-1, keepdim=True)+EPS)
                        rtvec_grad_rotnorm = rtvec_grad[:, :3]/(torch.norm(rtvec_grad[:, :3], dim=-1, keepdim=True)+EPS)
                        riem_grad_rot_loss = torch.mean(torch.sum((riem_grad_rotnorm - rtvec_grad_rotnorm)**2, dim=-1))
                        # riem_grad_rot_loss = -torch.mean(F.cosine_similarity(riem_grad_rotnorm, rtvec_grad_rotnorm, dim=-1))

                    if net == "miccai":
                        riem_grad_loss = riem_grad_trans_loss + riem_grad_rot_loss
                    else:
                        # Apr09 rockfish experiment setting
                        riem_grad_loss = 1.5*riem_grad_trans_loss + riem_grad_rot_loss + 0.5*riem_grad_transZ_loss

                riem_grad_loss.backward()

                # Clip training gradient magnitude
                torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
                optimizer.step()
                scheduler.step()

                total_loss = l2_loss

                mse_loss_               = torch.mean(l2_loss).detach().item()
                riem_grad_loss_         = riem_grad_loss.detach().item()
                riem_dist_mean_         = np.mean(riem_dist)
                total_loss_             = total_loss.detach().item()
                rtvec_grad_np           = rtvec_grad.detach().cpu().numpy()
                riem_grad_np            = riem_grad.detach().cpu().numpy()
                vecgrad_diff_           = rtvec_grad_np - riem_grad_np
                vecgrad_diff_norm_      = LA.norm(vecgrad_diff_, 2)
                rtvec_grad_rot_dist_    = F.cosine_similarity(rtvec_grad.detach().cpu()[:, :3], riem_grad.detach().cpu()[:, :3], dim=-1).numpy()
                rtvec_grad_trans_dist_  = F.cosine_similarity(rtvec_grad.detach().cpu()[:, 3:], riem_grad.detach().cpu()[:, 3:], dim=-1).numpy()

                mse_loss_list.append(mse_loss_)
                riem_grad_loss_list.append(riem_grad_loss_)
                rtvec_grad_rot_dist_list.append(rtvec_grad_rot_dist_)
                rtvec_grad_trans_dist_list.append(rtvec_grad_trans_dist_)

                riem_dist_list.append(riem_dist)
                riem_dist_mean_list.append(riem_dist_mean_)
                total_loss_list.append(total_loss_)
                vecgrad_diff_list.append(vecgrad_diff_)

                if not grad_geo_loss:
                    riem_grad_trans_loss_   = riem_grad_trans_loss.detach().item()
                    riem_grad_rot_loss_     = riem_grad_rot_loss.detach().item()
                    rtvec_grad_transnorm_   = rtvec_grad.detach().cpu().numpy()[:, 3:] if grad_mag_loss else rtvec_grad_transnorm.detach().cpu().numpy()
                    rtvec_grad_rotnorm_     = rtvec_grad.detach().cpu().numpy()[:, :3] if grad_mag_loss else rtvec_grad_rotnorm.detach().cpu().numpy()
                    riem_grad_transnorm_    = riem_grad.detach().cpu().numpy()[:, 3:] if grad_mag_loss else riem_grad_transnorm.detach().cpu().numpy()
                    riem_grad_rotnorm_      = riem_grad.detach().cpu().numpy()[:, :3] if grad_mag_loss else riem_grad_rotnorm.detach().cpu().numpy()
                    riem_grad_trans_loss_all_list.append(riem_grad_trans_loss_)
                    riem_grad_rot_loss_all_list.append(riem_grad_rot_loss_)
                    rtvec_grad_trans_norm_list.append(rtvec_grad_transnorm_)
                    rtvec_grad_rot_norm_list.append(rtvec_grad_rotnorm_)
                    riem_grad_trans_norm_list.append(riem_grad_transnorm_)
                    riem_grad_rot_norm_list.append(riem_grad_rotnorm_)
                else:
                    riem_grad_loss_all_list.append(riem_grad_loss_)
                    rtvec_grad_transnorm_   = rtvec_grad.detach().cpu().numpy()[:, 3:]
                    rtvec_grad_rotnorm_     = rtvec_grad.detach().cpu().numpy()[:, :3]
                    riem_grad_transnorm_    = riem_grad.detach().cpu().numpy()[:, 3:]
                    riem_grad_rotnorm_      = riem_grad.detach().cpu().numpy()[:, :3]
                    rtvec_grad_trans_norm_list.append(rtvec_grad_transnorm_)
                    rtvec_grad_rot_norm_list.append(rtvec_grad_rotnorm_)
                    riem_grad_trans_norm_list.append(riem_grad_transnorm_)
                    riem_grad_rot_norm_list.append(riem_grad_rotnorm_)

                torch.cuda.empty_cache()

                cur_lr = float(scheduler.get_lr()[0])

                print('Train epoch: {} Iter: {} RegiSim: {:.4f}, gLoss: {:.4f} ± {:.2f}, LR: {:.4f} * 10^-5'.format(
                            epoch, iter, np.mean(total_loss_list), np.mean(riem_grad_loss_list), np.std(riem_grad_loss_list),
                            cur_lr * 100000, sys.stdout))

                if writing_to_disk and log_tensorboard_writer:
                    writer.add_scalar('Network Similarity', total_loss_, step_cnt)
                    writer.add_scalar('Riem Grad Loss', riem_grad_loss_, step_cnt)
                    writer.add_scalar('Riem Grad Trans Loss', riem_grad_trans_loss_, step_cnt)
                    writer.add_scalar('Riem Grad Rot Loss', riem_grad_rot_loss_, step_cnt)
                    writer.add_scalar('Riem Dist Mean', riem_dist_mean_, step_cnt)
                    writer.add_scalar('Vec Grad Diff', vecgrad_diff_norm_, step_cnt)

        if writing_to_disk:
            if epoch % LOSS_INFO_EVERY_EPOCH == 0:
                if geo_mag_loss:
                    plot_geomag_run_stat(stat_figs_folder, epoch, riem_dist_list, mse_loss_diff_list, riem_dist_loss_list)
                elif grad_geo_loss:
                    plot_run_stat_geo(stat_figs_folder, epoch, riem_dist_list, riem_dist_mean_list, mse_loss_list,
                                      riem_grad_loss_list, vecgrad_diff_list, rtvec_grad_rot_dist_list, rtvec_grad_trans_dist_list,
                                      rtvec_grad_trans_norm_list, rtvec_grad_rot_norm_list, riem_grad_trans_norm_list, riem_grad_rot_norm_list,
                                      riem_grad_loss_all_list, training_loss_ylim=training_loss_ylim)
                else:
                    plot_run_stat_mse(stat_figs_folder, epoch, riem_dist_list, riem_dist_mean_list, mse_loss_list,
                                      riem_grad_loss_list, vecgrad_diff_list, rtvec_grad_rot_dist_list, rtvec_grad_trans_dist_list,
                                      rtvec_grad_trans_norm_list, rtvec_grad_rot_norm_list, riem_grad_trans_norm_list, riem_grad_rot_norm_list,
                                      riem_grad_trans_loss_all_list, riem_grad_rot_loss_all_list, training_loss_ylim=training_loss_ylim)

        # Clear debugging info list
        if epoch % EPOCH_LOSS_CACHE == 0:
            riem_dist_list = []

            if not geo_mag_loss:
                mse_loss_list = []
                riem_dist_mean_list = []
                riem_grad_loss_list = []
                vecgrad_diff_list = []
                rtvec_grad_trans_norm_list = []
                rtvec_grad_rot_norm_list = []
                riem_grad_trans_norm_list = []
                riem_grad_rot_norm_list = []
                rtvec_grad_rot_dist_list = []
                rtvec_grad_trans_dist_list = []
            else:
                mse_loss_diff_list = []

        def save_net(net_path):
            tmp_name = '{}.tmp'.format(net_path)
            torch.save({ 'epoch'                : epoch,
                         'model-state-dict'     : model.state_dict(),
                         'optimizer-state-dict' : optimizer.state_dict(),
                         'scheduler-state-dict' : scheduler.state_dict(),
                         'save-path'            : SAVE_PATH,
                         'h5-file'              : H5_File,
                         'grid-step-size'       : STEP_SIZE,
                         'iter-num'             : ITER_NUM,
                         'end-epoch'            : END_EPOCH,
                         'save-freq'            : SAVE_MODEL_EVERY_EPOCH,
                         'val-freq'             : VALID_SIM_LOSS_EPOCH,
                         'debug-plot-freq'      : SAVE_DEBUG_PLOT_EVERY_ITER,
                         'norm-ct'              : NORM_CT,
                         'norm-mov'             : NORM_MOV,
                         'no-3d-ori'            : NO_3D_ORI,
                         'no-3d-net'            : NO_3D_NET,
                         'debug-plot'           : debug_plot,
                         'writing-to-disk'      : writing_to_disk,
                         'log-nan-tensor'       : log_nan_tensor,
                         'tensorboard-writer'   : log_tensorboard_writer,
                         'valid-sim-loss'       : valid_sim_loss,
                         'valid-ct-id'          : val_ct_id,
                         'valid-proj-id'        : val_proj_id,
                         'use-fixed-grid'       : use_fixed_grid,
                         'batch-size'           : BATCH_SIZE,
                         'data-aug'             : do_aug,
                         'geo-mag-loss'         : geo_mag_loss,
                         'grad-mag-loss'        : grad_mag_loss,
                         'grad-geo-loss'        : grad_geo_loss,
                         'same-batch-target'    : same_batch_target,
                         'random-state'         : random.getstate(),
                         'numpy-random-state'   : np.random.get_state(),
                         'torch-random-state'   : torch.get_rng_state(),
                         'cuda-random-state'    : torch.cuda.get_rng_state()
                          },
                       tmp_name)
            shutil.move(tmp_name, net_path)

        if writing_to_disk and epoch % SAVE_MODEL_EVERY_EPOCH == 0:
            checkpoint_filename = checkpoint_folder + '/vali_model' + str(epoch) + '.pt'
            save_net(checkpoint_filename)

        # Due to memory limitation, this is not used for now
        if valid_sim_loss and epoch % VALID_SIM_LOSS_EPOCH == 0:
            print('Running validation...')
            model.eval()
            model.require_grad = False

            val_network_sim_list_allDOF = list()

            with torch.no_grad():
                for offsetDOF in range(6):
                    if offsetDOF in [0, 1, 2]:
                        offsetmag = valid_offset_rot * DEG2RAD
                    elif offsetDOF in [3, 4]:
                        offsetmag = valid_offset_trans / norm_factor
                    elif offsetDOF in [5]:
                        offsetmag = valid_transZoffset_scale * valid_offset_trans / norm_factor
                    else:
                        sys.exit("offsetDOF is not in range!")

                    # Preprocess offset settings
                    val_network_sim_list = []
                    val_gradncc_sim_list = []

                    euler_rtvec_smp = np.zeros((1, 6))
                    for idz in range(valid_num_sample):
                        euler_rtvec_smp[0, offsetDOF] = -offsetmag + idz*(2*offsetmag)/valid_num_sample
                        smp_mat4x4 = convert_numpy_euler_rtvec_to_mat4x4(euler_rtvec_smp, device)
                        mat4x4 = torch.matmul(smp_mat4x4, transform_mat4x4_val_gt)

                        with torch.no_grad():
                            rtvec_val = convert_transform_mat4x4_to_rtvec(mat4x4)

                        rtvec_val.requires_grad = True

                        # Do Projection
                        vals = model(val_3D_vol, target_valid, rtvec_val, val_corner_pt, val_param, log_nan_tensor=log_nan_tensor) if not use_fixed_grid \
                            else model(val_3D_vol, target_valid, rtvec_val, val_grid, log_nan_tensor=log_nan_tensor)

                        if not encoder_share_weights:
                            encode_mov = vals[0]
                            encode_tar = vals[1]

                            # Calculate Net l2 Loss, L_N
                            l2_loss = criterion_mse(encode_mov, encode_tar) #RegiNet loss
                        else:
                            encode_out = vals[0]
                            l2_loss = torch.mean(encode_out)

                        val_network_sim_list.append(l2_loss.item())

                    val_network_sim_list_allDOF.append(val_network_sim_list)

                fig_val, axes = plt.subplots(3, 2, figsize=(40,25))
                fig_val.suptitle(str(VAL_CT_NAME) + " Validation Epoch " + str(epoch))

                for offsetDOF in range(6):
                    if offsetDOF in [0, 1, 2]:
                        offsetmag = valid_offset_rot * DEG2RAD
                        ax_x = offsetDOF
                        ax_y = 0
                    elif offsetDOF in [3, 4]:
                        offsetmag = valid_offset_trans / norm_factor
                        ax_x = offsetDOF - 3
                        ax_y = 1
                    elif offsetDOF in [5]:
                        offsetmag = valid_transZoffset_scale * valid_offset_trans / norm_factor
                        ax_x = offsetDOF - 3
                        ax_y = 1
                    else:
                        sys.exit("offsetDOF is not in range!")

                    xdim = val_xdim_list_allDOF[offsetDOF]
                    network_sim_list = val_network_sim_list_allDOF[offsetDOF]
                    gradncc_sim_list = val_gradncc_sim_list_allDOF[offsetDOF]
                    network_sim_arr = np.array(network_sim_list)
                    gradncc_sim_arr = np.array(gradncc_sim_list)
                    if not no_valid_sim_norm:
                        eps = 1.0e-6
                        network_sim_arr = (network_sim_arr - np.min(network_sim_arr) + eps) / (np.max(network_sim_arr) - np.min(network_sim_arr) + eps)
                        gradncc_sim_arr = (gradncc_sim_arr - np.min(gradncc_sim_arr) + eps) / (np.max(gradncc_sim_arr) - np.min(gradncc_sim_arr) + eps)

                    axes[ax_x, ax_y].plot(xdim, network_sim_arr, 'bo-')
                    if not no_valid_sim_norm:
                        axes[ax_x, ax_y].plot(xdim, gradncc_sim_arr, 'ro-')

                    axes[ax_x, ax_y].set_xlabel(plots_xlabel_txt[offsetDOF])
                    axes[ax_x, ax_y].set_ylabel('Similarity')
                    if not no_valid_sim_norm:
                        axes[ax_x, ax_y].legend(['Network Similarity', 'GradNCC Similarity'])
                    else:
                        axes[ax_x, ax_y].legend(['Network Similarity'])

                    axes[ax_x, ax_y].set_title(plots_zlabel_txt[offsetDOF])

                plt.show()
                plt.savefig(stat_figs_folder + '/Valid' + str(epoch) + '.png')
                plt.close(fig_val)

            torch.cuda.empty_cache()
