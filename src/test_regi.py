from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import numpy as np
from torch.optim.lr_scheduler import StepLR, CyclicLR

from module import RegiNet, ProST_init, Pelvis_Dataset
from util import gradncc, init_rtvec_test, input_param
from util_plot import plot_test_iter_comb

from geomstats.geometry.special_euclidean import SpecialEuclidean

device = torch.device("cuda")
PI = 3.1415926
NUM_PHOTON = 10000
BATCH_SIZE = 1
ITER_STEPS = 500

MANUAL_TEST = False

SE3_GROUP = SpecialEuclidean(n=3)
METRIC = SE3_GROUP.left_canonical_metric

CT_PATH = '../data/CT128.nii'
SEG_PATH = '../data/CTSeg128.nii'
VOX_SPAC = 2.33203125

SAVE_PATH = '../data/save_model'
RESUME_EPOCH = 90
lr_net = 0.01
lr_gradncc = 0.002
switch_trd = 0.003
stop_trd = 1e-4
zFlip = False


RESUME_MODEL = SAVE_PATH+'/pretrain.pt'

def train():
    criterion_mse = nn.MSELoss()
    criterion_gradncc = gradncc
    param, det_size, _3D_vol, CT_vol, ray_proj_mov, corner_pt, norm_factor = input_param(CT_PATH, SEG_PATH, BATCH_SIZE, VOX_SPAC, zFlip)

    initmodel = ProST_init(param).to(device)
    model = RegiNet(param, det_size).to(device)

    checkpoint = torch.load(RESUME_MODEL)
    model.load_state_dict(checkpoint['state_dict'])


    model.eval()
    model.require_grad = False

    fig = plt.figure(figsize=(15, 9))


     # Get target  projection
    if MANUAL_TEST:
        manual_rtvec_gt= np.array([[0,0,0,0,0,0]])
        manual_rtvec_smp= np.array([[-0.2, 0.3, 0.6, -0.1, 0.25, 0.2]])
    else:
        manual_rtvec_gt = None
        manual_rtvec_smp = None

    transform_mat3x4_gt, rtvec, rtvec_gt = init_rtvec_test(device, manual_test=MANUAL_TEST,
                                                           manual_rtvec_gt=manual_rtvec_gt,
                                                           manual_rtvec_smp=manual_rtvec_smp)

    with torch.no_grad():
        target = initmodel(CT_vol, ray_proj_mov, transform_mat3x4_gt, corner_pt)
        min_tar, _ = torch.min(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
        max_tar, _ = torch.max(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
        target = (target.reshape(BATCH_SIZE, -1) - min_tar) / (max_tar - min_tar)
        target = target.reshape(BATCH_SIZE, 1, det_size, det_size)


    optimizer_net = optim.SGD([rtvec], lr=lr_net, momentum=0.9)
    optimizer_gradncc = optim.SGD([rtvec], lr=lr_gradncc, momentum=0.9)
    #scheduler_net = CyclicLR(optimizer_gradncc, base_lr=0.005, max_lr=0.02,step_size_up=20)
    scheduler_gradncc = CyclicLR(optimizer_gradncc, base_lr=0.001, max_lr=0.003,step_size_up=20)

    network_sim_list = []
    gradncc_sim_list = []
    rtvec_diff_list = []

    stop = False
    switch = False
    for iter in range(ITER_STEPS):
        if not switch:
            if iter > 10:
                network_sim_list_np = np.array(network_sim_list)
                switch = np.std(network_sim_list_np[-10:]) < switch_trd
            else:
                switch = False

        if switch and iter>10:
            stop = np.std(gradncc_sim_list[-10:]) < stop_trd

        if stop:
            break

        # Do Projection
        encode_mov, encode_tar, proj_mov = model(_3D_vol, target, rtvec, corner_pt)

        optimizer_net.zero_grad()
        optimizer_gradncc.zero_grad()

        # Network Similarity:
        l2_loss = criterion_mse(encode_mov, encode_tar)
        # gradncc Similarity:
        gradncc_loss = criterion_gradncc(proj_mov, target)

        network_sim_list.append(l2_loss.item())
        gradncc_sim_list.append(gradncc_loss.item())

        rtvec.retain_grad()

        if switch:
            gradncc_loss.backward()
            scheduler_gradncc.step()
        else:
            l2_loss.backward()
            #scheduler_net.step()

        rtvec_diff = rtvec.detach().cpu()[0,:]-rtvec_gt.detach().cpu()[0,:]
        rtvec_diff_list.append(rtvec_diff.detach().cpu().numpy())

        if switch:
            optimizer_gradncc.step()
        else:
            optimizer_net.step()

        if iter == 0:
            proj_init_numpy0 = np.array(proj_mov[0,0,:,:].data.cpu())

        plot_test_iter_comb(fig, proj_mov, proj_init_numpy0, target, det_size, norm_factor,\
               network_sim_list, gradncc_sim_list, rtvec_diff_list, switch)



if __name__ == "__main__":
    train()
