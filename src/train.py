from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

import sys
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, CyclicLR

from module import RegiNet, Pelvis_Dataset, ProST_init
from util import ncc, input_param, count_parameters, init_rtvec_train

from geomstats.geometry.special_euclidean import SpecialEuclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric
import geomstats.geometry.riemannian_metric as riem

device = torch.device("cuda")
PI = 3.1415926
NUM_PHOTON = 20000
BATCH_SIZE = 2
EPS = 1e-10
ITER_NUM = 200
clipping_value = 10
SAVE_MODEL_EVERY_EPOCH = 5

SE3_GROUP = SpecialEuclidean(n=3, point_type='vector')
RiemMetric = RiemannianMetric(dim=6)
METRIC = SE3_GROUP.left_canonical_metric
riem_dist_fun = RiemMetric.dist

CT_PATH = '../data/CT128.nii'
SEG_PATH = '../data/CTSeg128.nii'
SAVE_PATH = '../data/save_model'
VOX_SPAC = 2.33203125

RESUME_EPOCH = -1 #-1 means training from scratch
RESUME_MODEL = SAVE_PATH+'/checkpoint/vali_model'+str(RESUME_EPOCH)+'.pt'

zFlip = False

def train():
    criterion_mse = nn.MSELoss()

    param, det_size, _3D_vol, CT_vol, ray_proj_mov, corner_pt, norm_factor = input_param(CT_PATH, SEG_PATH, BATCH_SIZE, VOX_SPAC, zFlip)

    initmodel = ProST_init(param).to(device)
    model = RegiNet(param, det_size).to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    scheduler = CyclicLR(optimizer, base_lr=1e-6, max_lr=1e-4,step_size_up=100)

    if RESUME_EPOCH>=0:
        print('Resuming model from epoch', RESUME_EPOCH)
        checkpoint = torch.load(RESUME_MODEL)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        START_EPOCH = RESUME_EPOCH + 1
        step_cnt = RESUME_EPOCH*ITER_NUM
    else:
        START_EPOCH = 0
        step_cnt = 0

    print('module parameters:', count_parameters(model))


    model.train()

    riem_grad_loss_list = []
    riem_grad_rot_loss_list = []
    riem_grad_trans_loss_list = []
    riem_dist_list = []
    riem_dist_mean_list = []
    mse_loss_list = []
    vecgrad_diff_list = []
    total_loss_list = []

    for epoch in range(START_EPOCH, 20000):
        ## Do Iterative Validation
        model.train()
        for iter in tqdm(range(ITER_NUM)):
            step_cnt = step_cnt+1
            scheduler.step()
            # Get target  projection
            transform_mat3x4_gt, rtvec, rtvec_gt = init_rtvec_train(BATCH_SIZE, device)

            with torch.no_grad():
                target = initmodel(CT_vol, ray_proj_mov, transform_mat3x4_gt, corner_pt)
                min_tar, _ = torch.min(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                max_tar, _ = torch.max(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
                target = (target.reshape(BATCH_SIZE, -1) - min_tar) / (max_tar - min_tar)
                target = target.reshape(BATCH_SIZE, 1, det_size, det_size)

            # Do Projection and get two encodings
            encode_mov, encode_tar, proj_mov = model(_3D_vol, target, rtvec, corner_pt)

            optimizer.zero_grad()
            # Calculate Net l2 Loss, L_N
            l2_loss = criterion_mse(encode_mov, encode_tar)

            # Find geodesic distance
            riem_dist = np.sqrt(riem.loss(rtvec.detach().cpu(), rtvec_gt.detach().cpu(), METRIC))

            z = Variable(torch.ones(l2_loss.shape)).to(device)
            rtvec_grad = torch.autograd.grad(l2_loss, rtvec, grad_outputs=z, only_inputs=True, create_graph=True,
                                                  retain_graph=True)[0]
            # Find geodesic gradient
            riem_grad = riem.grad(rtvec.detach().cpu(), rtvec_gt.detach().cpu(), METRIC)
            riem_grad = torch.tensor(riem_grad, dtype=torch.float, requires_grad=False, device=device)

            ### Translation Loss
            riem_grad_transnorm = riem_grad[:, 3:]/(torch.norm(riem_grad[:, 3:], dim=-1, keepdim=True)+EPS)
            rtvec_grad_transnorm = rtvec_grad[:, 3:]/(torch.norm(rtvec_grad[:, 3:], dim=-1, keepdim=True)+EPS)
            riem_grad_trans_loss = torch.mean(torch.sum((riem_grad_transnorm - rtvec_grad_transnorm)**2, dim=-1))

            ### Rotation Loss
            riem_grad_rotnorm = riem_grad[:, :3]/(torch.norm(riem_grad[:, :3], dim=-1, keepdim=True)+EPS)
            rtvec_grad_rotnorm = rtvec_grad[:, :3]/(torch.norm(rtvec_grad[:, :3], dim=-1, keepdim=True)+EPS)
            riem_grad_rot_loss = torch.mean(torch.sum((riem_grad_rotnorm - rtvec_grad_rotnorm)**2, dim=-1))

            riem_grad_loss = riem_grad_trans_loss + riem_grad_rot_loss

            riem_grad_loss.backward()

            # Clip training gradient magnitude
            torch.nn.utils.clip_grad_norm(model.parameters(), clipping_value)
            optimizer.step()

            total_loss = l2_loss

            mse_loss_list.append(torch.mean(l2_loss).detach().item())
            riem_grad_loss_list.append(riem_grad_loss.detach().item())
            riem_grad_rot_loss_list.append(riem_grad_rot_loss.detach().item())
            riem_grad_trans_loss_list.append(riem_grad_trans_loss.detach().item())
            riem_dist_list.append(riem_dist)
            riem_dist_mean_list.append(np.mean(riem_dist))
            total_loss_list.append(total_loss.detach().item())
            vecgrad_diff = (rtvec_grad - riem_grad).detach().cpu().numpy()
            vecgrad_diff_list.append(vecgrad_diff)

            torch.cuda.empty_cache()

            cur_lr = float(scheduler.get_lr()[0])

            print('Train epoch: {} Iter: {} tLoss: {:.4f}, gLoss: {:.4f}/{:.2f}, gLoss_rot: {:.4f}/{:.2f}, gLoss_trans: {:.4f}/{:.2f}, LR: {:.4f}'.format(
                        epoch, iter, np.mean(total_loss_list), np.mean(riem_grad_loss_list), np.std(riem_grad_loss_list),\
                                     np.mean(riem_grad_rot_loss_list), np.std(riem_grad_rot_loss_list),\
                                     np.mean(riem_grad_trans_loss_list), np.std(riem_grad_trans_loss_list),
                        cur_lr, sys.stdout))


        if epoch%SAVE_MODEL_EVERY_EPOCH == 0:
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state, SAVE_PATH + '/checkpoint/vali_model' + str(epoch) + '.pt')

        riem_grad_loss_list = []
        riem_grad_rot_loss_list = []
        riem_grad_trans_loss_list = []
        riem_dist_list = []
        riem_dist_mean_list = []
        mse_loss_list = []
        vecgrad_diff_list = []


if __name__ == "__main__":
    train()
