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

from module_vit import RegiNet_CrossViTv2_SW
from posevec2mat import euler2mat
from util import gradncc, count_parameters, norm_target, generate_fixed_grid, convert_numpy_euler_rtvec_to_mat4x4, \
                 convert_transform_mat4x4_to_rtvec
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
    parser.add_argument('--iter-fig-freq', help='Number of iterations to save a debug plot figure to disk folder', type=int, default=50)
    parser.add_argument('-n', '--no-writing-to-disk', help='Not creating folders or writing anything to disk', action='store_true', default=False)
    parser.add_argument('--cuda-id', help='Specify CUDA id when training on pong', type=str, default="")

    args = parser.parse_args()

    SAVE_PATH = args.save_path
    H5_File = args.h5_file
    RESUME_EPOCH = args.resume_epoch
    BATCH_SIZE = args.batch_size
    STEP_SIZE = args.step_size
    ITER_NUM = args.iter_num
    END_EPOCH = args.epoch_num
    SAVE_MODEL_EVERY_EPOCH = args.save_freq
    writing_to_disk = not args.no_writing_to_disk
    cuda_id = args.cuda_id

    # This is specifically to run on pong
    if cuda_id:
        os.environ["CUDA_VISIBLE_DEVICES"]=cuda_id

    checkpoint_folder = SAVE_PATH + '/checkpoint'
    resumed_checkpoint_filename = checkpoint_folder + '/vali_model'+str(RESUME_EPOCH)+'.pt'

    log = logging.getLogger('deepdrr')
    log.propagate = False

    START_EPOCH = 0
    step_cnt = 0

    if writing_to_disk:
        if not os.path.exists(SAVE_PATH):
            print('Creating...' + SAVE_PATH)
            os.mkdir(SAVE_PATH)

        if not os.path.exists(checkpoint_folder):
            print('Creating...' + checkpoint_folder)
            os.mkdir(checkpoint_folder)

    hf = h5py.File(H5_File, 'r')

    CT_DATA_LIST = hf.get("ct-list")
    num_ct = len(CT_DATA_LIST)
    print("H5 file loaded:\n", H5_File)
    print("Number of CTs:\n", num_ct)

    # Training Setup
    criterion_mse = nn.MSELoss()
    NORM_MOV = False
    model = RegiNet_CrossViTv2_SW().to(device)
    encoder_share_weights = True
    
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    scheduler = CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-4,step_size_up=100)

    if RESUME_EPOCH >= 0:
        print('Resuming model from', resumed_checkpoint_filename)
        model.load_state_dict(prev_state['model-state-dict'])
        optimizer.load_state_dict(prev_state['optimizer-state-dict'])
        scheduler.load_state_dict(prev_state['scheduler-state-dict'])

    print('module parameters:', count_parameters(model))

    # Start -- Generate validation target image
    riem_dist_list = []

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

        _3D_vol_np = (_3D_vol_np - np.min(_3D_vol_np))/(np.max(_3D_vol_np) - np.min(_3D_vol_np))
        _3D_vol = torch.tensor(np.repeat(_3D_vol_np, BATCH_SIZE, axis=0), dtype=torch.float, requires_grad=True, device=device)
        CT_vol  = torch.tensor(np.repeat(CT_vol_np, BATCH_SIZE, axis=0), dtype=torch.float, requires_grad=True, device=device)
        ray_proj_mov = torch.tensor(np.repeat(ray_proj_mov_np, BATCH_SIZE, axis=0), dtype=torch.float, requires_grad=True, device=device)
        corner_pt = torch.tensor(np.repeat(corner_pt_np, BATCH_SIZE, axis=0), dtype=torch.float, requires_grad=True, device=device)

        for iter in range(ITER_NUM):
            step_cnt = step_cnt+1

            # Load target image from h5 file
            rtvec_gt_arr = np.zeros((BATCH_SIZE, 6))
            euler_rtvec_smp_arr = np.zeros((BATCH_SIZE, 6))
            euler_rtvec_gt_arr = np.zeros((BATCH_SIZE, 6))
            target_arr = np.zeros((BATCH_SIZE, 128, 128))

            for b in range(BATCH_SIZE):
                target_proj_ID = random.randint(0, num_proj-1)
                target_proj_name = str(target_proj_ID).zfill(5)
                proj_grp = ct_grp[target_proj_name]

                rtvec_gt_arr[b, :] = np.array(proj_grp.get("angvec"))
                euler_rtvec_gt_arr[b, :] = np.array(proj_grp.get("eulervec"))
                target_arr[b, :, :] = np.array(proj_grp.get("proj"))

            rtvec, rtvec_gt = rand_smp_ang_rtvec(rtvec_gt_arr, norm_factor)

            with torch.no_grad():
                target = torch.tensor(target_arr, dtype=torch.float, requires_grad=True, device=device)
                target = norm_target(BATCH_SIZE, det_size, target)

            # Data augmentation
            target = aug_torch_target(target, device)

            # Do Projection and get two encodings
            vals = model(_3D_vol, target, rtvec, corner_pt, param)

            if type(vals) == bool and (not vals):# and (not vals[0])
                continue
            elif len(vals) == 3 and vals[-1] and (not encoder_share_weights):
                encode_mov = vals[0]
                encode_tar = vals[1]
            elif len(vals) == 2 and vals[-1] and encoder_share_weights:
                encode_out = vals[0]
            else:
                print('Invalid Model Return!!!!!')
                continue

            optimizer.zero_grad()

            # Find geodesic distance
            riem_dist = np.sqrt(riem.loss(rtvec.detach().cpu(), rtvec_gt.detach().cpu(), METRIC))

            # Calculate Net l2 Loss, L_N
            l2_loss = torch.mean(encode_out) if encoder_share_weights else criterion_mse(encode_mov, encode_tar) #RegiNet loss

            z = Variable(torch.ones(l2_loss.shape)).to(device)

            rtvec_grad = torch.autograd.grad(l2_loss, rtvec, grad_outputs=z, only_inputs=True, create_graph=True,
                                                    retain_graph=True)[0]

            # Find geodesic gradient
            riem_grad = riem.grad(rtvec.detach().cpu(), rtvec_gt.detach().cpu(), METRIC)
            riem_grad = torch.tensor(riem_grad, dtype=torch.float, requires_grad=False, device=device)

            # Training loss is defined as geodesic loss between two gradient vectors (M_dist defined in the paper)
            riem_grad_loss = Geodesic_loss.apply(rtvec_grad, riem_grad)
        
            riem_grad_loss.backward()

            # Clip training gradient magnitude
            torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
            optimizer.step()
            scheduler.step()

            torch.cuda.empty_cache()

            cur_lr = float(scheduler.get_lr()[0])

            print('Train epoch: {} Iter: {}, LR: {:.4f} * 10^-5'.format(
                        epoch, iter, cur_lr * 100000, sys.stdout))

        def save_net(net_path):
            tmp_name = '{}.tmp'.format(net_path)
            torch.save({ 'model-state-dict'     : model.state_dict(),
                         'optimizer-state-dict' : optimizer.state_dict(),
                         'scheduler-state-dict' : scheduler.state_dict()
                          },
                       tmp_name)
            shutil.move(tmp_name, net_path)

        if writing_to_disk and epoch % SAVE_MODEL_EVERY_EPOCH == 0:
            checkpoint_filename = checkpoint_folder + '/prost_model' + str(epoch) + '.pt'
            save_net(checkpoint_filename)
