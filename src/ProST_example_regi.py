from __future__ import print_function
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from util import ncc, input_param, init_rtvec_test, gradncc
from util_plot import plot_example_regi
from module import ProST

device = torch.device("cuda")
PI = 3.1415926
NUM_PHOTON = 10000
ITER_STEPS = 100

CT_PATH = '../data/CT128.nii'
SEG_PATH = '../data/CTSeg128.nii'
BATCH_SIZE = 1

def main():
    # Use Gradient-NCC similarity as loss function
    criterion_gradncc = gradncc
    # Calculate geometric parameters
    param, det_size, _3D_vol, CT_vol, ray_proj_mov, corner_pt, norm_factor = input_param(CT_PATH, SEG_PATH, BATCH_SIZE)
    # Initialize projection model
    projmodel = ProST(param).to(device)

    ########## Hard Code test groundtruth and initialize poses ##########
    # [rx, ry, rz, tx, ty, tz]
    manual_rtvec_gt= np.array([[0, 0, 0, 0, 0, 0]])
    manual_rtvec_smp= np.array([[0.0, 0.0, 5.0, 5.0, 0.0, 0.0]])

    # Normalization and conversion to transformation matrix
    manual_rtvec_smp[:, :3] = manual_rtvec_smp[:, :3]*PI/180
    manual_rtvec_smp[:, 3:] = manual_rtvec_smp[:, 3:]/norm_factor
    transform_mat3x4_gt, rtvec, rtvec_gt = init_rtvec_test(device, manual_test=True,
                                                           manual_rtvec_gt=manual_rtvec_gt,
                                                           manual_rtvec_smp=manual_rtvec_smp)
    with torch.no_grad():
        target = projmodel(_3D_vol, ray_proj_mov, rtvec_gt, corner_pt)
        # Min-Max to [0,1] normalization for target image
        min_tar, _ = torch.min(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
        max_tar, _ = torch.max(target.reshape(BATCH_SIZE, -1), dim=-1, keepdim=True)
        target = (target.reshape(BATCH_SIZE, -1) - min_tar) / (max_tar - min_tar)
        target = target.reshape(BATCH_SIZE, 1, det_size, det_size)
        plt.imshow(target.detach().cpu().numpy()[0,0,:,:].reshape(det_size, det_size), cmap='gray')
        plt.title('target img')
        plt.show()

    # Use Pytorch SGD optimizer
    optimizer_gradncc = optim.SGD([rtvec], lr=0.002, momentum=0.9)
    gradncc_sim_list = []
    rtvec_diff_list = []
    fig = plt.figure(figsize=(15, 9))
    for iter in range(ITER_STEPS):
        rtvec_diff = rtvec.detach().cpu()[0,:]-rtvec_gt.detach().cpu()[0,:]
        rtvec_diff_list.append(rtvec_diff.detach().cpu().numpy())
        # Do Projection
        proj_mov = projmodel(_3D_vol, target, rtvec, corner_pt)

        optimizer_gradncc.zero_grad()

        # GradNCC Similarity:
        gradncc_loss = criterion_gradncc(proj_mov, target)

        gradncc_sim_list.append(gradncc_loss.item())

        gradncc_loss.backward()

        optimizer_gradncc.step()

        if iter == 0:
            proj_init_numpy0 = np.array(proj_mov[0,0,:,:].data.cpu())

        plot_example_regi(fig, proj_mov, proj_init_numpy0, target, det_size, norm_factor, gradncc_sim_list, rtvec_diff_list)

if __name__ == "__main__":
    main()
