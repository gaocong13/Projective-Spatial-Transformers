import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import glob

PI = 3.1415926

def create_video_from_figs(SAVE_PATH, TEST_ID):
    iterfig_dir = SAVE_PATH + '/movie/iterfigs_' + TEST_ID
    img_array = []
    for filename in sorted(glob.glob(iterfig_dir+'/*.jpg')):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    movie_name = SAVE_PATH + '/movie/animation_' + TEST_ID + '.mp4'
    out = cv2.VideoWriter(movie_name,cv2.VideoWriter_fourcc(*'mp4v'), 1, size)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()

def plot_run_stat(SAVE_PATH, epoch, riem_dist_list, riem_dist_mean_list, mse_loss_list, riem_grad_loss_list, vecgrad_diff_list):

    riem_dist_array = np.array(riem_dist_list).reshape(-1, 1)
    riem_dist_mean_array = np.array(riem_dist_mean_list)
    mse_loss_array = np.array(mse_loss_list)
    riem_grad_loss_array = np.array(riem_grad_loss_list)
    vecgrad_diff_array = np.array(vecgrad_diff_list).reshape(-1, 6)

    fig = plt.figure(figsize=(20,5))
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)

    ax1.plot(riem_dist_mean_array, riem_grad_loss_array, 'ro')
    ax1.set_title('Gradient Loss')
    ax1.set_xlabel('Geodesic dist')
    ax1.set_ylabel('Grad Loss')
    ax1.axis([0, 1.2, 0.0, 4.0])
    ax1.grid(True)

    ax2.plot(riem_dist_mean_array, mse_loss_array, 'bo')
    #ax2.plot(riem_dist_mean_array[--200:], riem_dist_mean_array[--200:], 'co')
    ax2.set_title('Network Similarity - Geo Dist')
    ax2.set_xlabel('Geodesic dist')
    ax2.set_ylabel('NetSim')
    ax2.grid(True)

    ax3.plot(riem_dist_array, vecgrad_diff_array[:, :3], 'o')
    ax3.legend(['rx', 'ry', 'rz'])
    ax3.set_title('Rot Grad diff')
    ax3.axis([0, 1.2, -1.5, 1.5])
    ax3.grid(True)

    ax4.plot(riem_dist_array, vecgrad_diff_array[:, 3:], 'o')
    ax4.set_xlabel('Geodesic dist')
    ax4.legend(['tx', 'ty', 'tz'])
    ax4.set_title('Trans Grad diff')
    ax4.axis([0, 1.2, -2, 2])
    ax4.grid(True)

    plt.savefig(SAVE_PATH + '/stat_figs/Fig'+str(epoch)+'_loss.png')

    fig = plt.figure(figsize=(20,5))
    ax1 = fig.add_subplot(161)
    ax2 = fig.add_subplot(162)
    ax3 = fig.add_subplot(163)
    ax4 = fig.add_subplot(164)
    ax5 = fig.add_subplot(165)
    ax6 = fig.add_subplot(166)

    ax1.hist(vecgrad_diff_array[:, 0], bins=30, density=True)
    ax1.set_title('ax')
    ax1.set_xlim(-1.5, 1.5)
    ax1.grid(True)

    ax2.hist(vecgrad_diff_array[:, 1], bins=30, density=True)
    ax2.set_title('ay')
    ax2.set_xlim(-1.5, 1.5)
    ax2.grid(True)

    ax3.hist(vecgrad_diff_array[:, 2], bins=30, density=True)
    ax3.set_title('az')
    ax3.set_xlim(-1, 1)
    ax3.grid(True)

    ax4.hist(vecgrad_diff_array[:, 3], bins=30, density=True)
    ax4.set_title('tx')
    ax4.set_xlim(-1.5, 1.5)
    ax4.grid(True)

    ax5.hist(vecgrad_diff_array[:, 4], bins=30, density=True)
    ax5.set_title('ty')
    ax5.set_xlim(-1.5, 1.5)
    ax5.grid(True)

    ax6.hist(vecgrad_diff_array[:, 5], bins=30, density=True)
    ax6.set_title('tz')
    ax6.set_xlim(-2.5, 2.5)
    ax6.grid(True)

    plt.savefig(SAVE_PATH + '/stat_figs/Fig'+str(epoch)+'_hist.png')

def plot_vali_stat(SAVE_PATH, epoch, vecgrad_diff_list, riem_dist_list):
    vecgrad_diff_array = np.array(vecgrad_diff_list).reshape(-1, 6)
    riem_dist_array = np.array(riem_dist_list).reshape(-1, 1)

    fig = plt.figure(figsize=(20,8))
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    ax1.plot(riem_dist_array, vecgrad_diff_array[:, 0], 'ro')
    ax1.set_title('ax')
    ax1.axis([0, 1.5, -1.5, 1.5])
    ax1.grid(True)

    ax2.plot(riem_dist_array, vecgrad_diff_array[:, 1], 'go')
    ax2.set_title('ay')
    ax2.axis([0, 1.5, -1.5, 1.5])
    ax2.grid(True)

    ax3.plot(riem_dist_array, vecgrad_diff_array[:, 2], 'bo')
    ax3.set_title('az')
    ax3.axis([0, 1.5, -1.5, 1.5])
    ax3.grid(True)

    ax4.plot(riem_dist_array, vecgrad_diff_array[:, 3], 'ro')
    ax4.set_title('tx')
    ax4.axis([0, 1.5, -2, 2])
    ax4.grid(True)

    ax5.plot(riem_dist_array, vecgrad_diff_array[:, 4], 'go')
    ax5.set_title('ty')
    ax5.axis([0, 1.5, -2, 2])
    ax5.grid(True)

    ax6.plot(riem_dist_array, vecgrad_diff_array[:, 5], 'bo')
    ax6.set_title('tz')
    ax6.axis([0, 1.5, -2, 2])
    ax6.grid(True)

    plt.savefig(SAVE_PATH + '/stat_figs/Fig'+str(epoch)+'_validation.png')

def plot_test_iter_comb(fig, proj_mov, proj_init_numpy0, target, det_size, norm_factor,\
                   network_sim_list, ncc_sim_list, rtvec_diff_list, switch=False):

    network_sim_list_np = np.array(network_sim_list)
    network_sim_STD = np.std(network_sim_list_np[-10:])
    ncc_sim_list_np = np.array(ncc_sim_list)
    ncc_sim_STD = np.std(ncc_sim_list_np[-10:])
    rtvec_diff_list_np = np.array(rtvec_diff_list)

    rtvec_diff_list_np[:, :3] = rtvec_diff_list_np[:, :3] * 180/PI
    rtvec_diff_list_np[:, 3:] = rtvec_diff_list_np[:, 3:] * norm_factor


    proj_mov_numpy0 = np.array(proj_mov[0,0,:,:].data.cpu())
    target_numpy0 = np.array((target[0,0,:,:]).data.cpu())
    proj_mov_numpy0 = proj_mov_numpy0.reshape((det_size , det_size))
    #proj_mov_grad = np.gradient(proj_mov_numpy0, edge_order=1)
    target_numpy0 = target_numpy0.reshape((det_size , det_size))
    #target_numpy0 = target_numpy0 + proj_mov_grad[0]*2

    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    ax1.imshow(proj_init_numpy0)
    ax1.set_title('Initial Proj')

    ax2.imshow(proj_mov_numpy0)
    ax2.set_title('Moving Proj')

    ax3.imshow(target_numpy0)
    ax3.set_title('Target')

    ax4.plot(rtvec_diff_list_np[:, 0], color='darkred', marker='o')
    ax4.plot(rtvec_diff_list_np[:, 1], color='red', marker='o')
    ax4.plot(rtvec_diff_list_np[:, 2], color='lightsalmon', marker='o')
    ax4.set_title('Rvec Diff')
    ax4.set_ylabel('degree')
    ax4.set_xlabel('iteration')
    ax4.tick_params(axis='y', labelcolor='tab:red')
    ax4.legend(['ax', 'ay', 'az'])
    ax4.grid()

    ax42 = ax4.twinx()
    ax42.plot(rtvec_diff_list_np[:, 3], color='navy', marker='^')
    ax42.plot(rtvec_diff_list_np[:, 4], color='blue', marker='^')
    ax42.plot(rtvec_diff_list_np[:, 5], color='deepskyblue', marker='^')
    ax42.set_ylabel('mm')
    ax42.legend(['x', 'y', 'z'])
    ax42.tick_params(axis='y', labelcolor='tab:blue')

    if switch:
        ax5.plot(ncc_sim_list_np, marker='o', color='red')
        ax5.set_title('NOW: NCC Similarity--STD:{:.6f}'.format(ncc_sim_STD))
    else:
        ax5.plot(ncc_sim_list_np, marker='o', color='blue')
        ax5.set_title('NCC Similarity--STD:{:.4f}'.format(ncc_sim_STD))

    ax5.set_xlabel('iteration')
    ax5.grid()

    if switch:
        ax6.plot(network_sim_list_np, marker='o', color='blue')
        ax6.set_title('Network Similarity--STD:{:.4f}'.format(network_sim_STD))
    else:
        ax6.plot(network_sim_list_np, marker='o', color='red')
        ax6.set_title('NOW: Network Similarity--STD:{:.4f}'.format(network_sim_STD))

    ax6.set_xlabel('iteration')
    ax6.grid()

    plt.show(block=False)
    plt.pause(0.5)
    plt.clf()

def plot_test_iter(fig, proj_mov, proj_init_numpy0, target, det_size, norm_factor,\
                   network_sim_list, rtvec_diff_list, rtvec_grad_list, riem_grad_list):
    riem_grad_list_np = np.array(riem_grad_list)
    rtvec_grad_list_np = np.array(rtvec_grad_list)
    network_sim_list_np = np.array(network_sim_list)
    rtvec_diff_list_np = np.array(rtvec_diff_list)

    rtvec_diff_list_np[:, :3] = rtvec_diff_list_np[:, :3] * 180/PI
    rtvec_diff_list_np[:, 3:] = rtvec_diff_list_np[:, 3:] * norm_factor


    proj_mov_numpy0 = np.array(proj_mov[0,0,:,:].data.cpu())
    target_numpy0 = np.array((target[0,0,:,:]).data.cpu())
    proj_mov_numpy0 = proj_mov_numpy0.reshape((det_size , det_size))
    target_numpy0 = target_numpy0.reshape((det_size , det_size))

    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    ax1.imshow(proj_init_numpy0)
    ax1.set_title('Initial Proj')

    ax2.imshow(proj_mov_numpy0)
    ax2.set_title('Moving Proj')

    ax3.imshow(target_numpy0)
    ax3.set_title('Target')

    ax4.plot(rtvec_diff_list_np[:, 0], color='darkred', marker='o')
    ax4.plot(rtvec_diff_list_np[:, 1], color='red', marker='o')
    ax4.plot(rtvec_diff_list_np[:, 2], color='lightsalmon', marker='o')
    ax4.set_title('Rvec Diff')
    ax4.set_ylabel('degree')
    ax4.set_xlabel('iteration')
    ax4.tick_params(axis='y', labelcolor='tab:red')
    ax4.legend(['ax', 'ay', 'az'])
    ax4.grid()

    ax42 = ax4.twinx()
    ax42.plot(rtvec_diff_list_np[:, 3], color='navy', marker='^')
    ax42.plot(rtvec_diff_list_np[:, 4], color='blue', marker='^')
    ax42.plot(rtvec_diff_list_np[:, 5], color='deepskyblue', marker='^')
    ax42.set_ylabel('mm')
    ax42.legend(['x', 'y', 'z'])
    ax42.tick_params(axis='y', labelcolor='tab:blue')


    ax5.plot(rtvec_grad_list_np[:, 5], marker='o')
    ax5.plot(riem_grad_list_np[:,5], marker='o')
    ax5.set_title('Z axis Grad')
    ax5.set_xlabel('iteration')
    ax5.legend(['Network Grad', 'Riem Grad'])
    ax5.grid()

    ax6.plot(network_sim_list_np, marker='o')
    ax6.set_title('Network Similarity')
    ax6.set_xlabel('iteration')
    ax6.grid()

    plt.show(block=False)
    plt.pause(1)
    plt.clf()

def plot_example_regi(fig, proj_mov, proj_init_numpy0, target, det_size, norm_factor,\
                   gradncc_sim_list, rtvec_diff_list):
    gradncc_sim_list_np = np.array(gradncc_sim_list)
    rtvec_diff_list_np = np.array(rtvec_diff_list)

    rtvec_diff_list_np[:, :3] = rtvec_diff_list_np[:, :3] * 180/PI
    rtvec_diff_list_np[:, 3:] = rtvec_diff_list_np[:, 3:] * norm_factor

    proj_mov_numpy0 = np.array(proj_mov[0,0,:,:].data.cpu())
    target_numpy0 = np.array((target[0,0,:,:]).data.cpu())
    proj_mov_numpy0 = proj_mov_numpy0.reshape((det_size , det_size))
    target_numpy0 = target_numpy0.reshape((det_size , det_size))

    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    ax1.imshow(proj_init_numpy0)
    ax1.set_title('Initial Proj')

    ax2.imshow(proj_mov_numpy0)
    ax2.set_title('Moving Proj')

    ax3.imshow(target_numpy0)
    ax3.set_title('Target')

    ax4.plot(rtvec_diff_list_np[:, 0], color='darkred', marker='o')
    ax4.plot(rtvec_diff_list_np[:, 1], color='red', marker='o')
    ax4.plot(rtvec_diff_list_np[:, 2], color='lightsalmon', marker='o')
    ax4.set_title('Rvec Diff')
    ax4.set_ylabel('degree')
    ax4.set_xlabel('iteration')
    ax4.tick_params(axis='y', labelcolor='tab:red')
    ax4.legend(['rx', 'ry', 'rz'])
    ax4.grid()

    ax5.plot(rtvec_diff_list_np[:, 3], color='navy', marker='^')
    ax5.plot(rtvec_diff_list_np[:, 4], color='blue', marker='^')
    ax5.plot(rtvec_diff_list_np[:, 5], color='deepskyblue', marker='^')
    ax5.set_ylabel('mm')
    ax5.legend(['x', 'y', 'z'])
    ax5.tick_params(axis='y', labelcolor='tab:blue')
    ax5.grid()

    ax6.plot(gradncc_sim_list_np, marker='o')
    ax6.set_title('GradNCC Similarity')
    ax6.set_xlabel('iteration')
    ax6.grid()

    plt.show(block=False)
    plt.pause(1)
    plt.clf()

def plot_realtest_iter(fig, proj_mov, proj_init_numpy0, target, det_size, norm_factor,\
                   network_sim_list, rtvec_diff_list, rtvec_grad_list, riem_grad_list):
    riem_grad_list_np = np.array(riem_grad_list)
    rtvec_grad_list_np = np.array(rtvec_grad_list)
    rtvec_diff_list_np = np.array(rtvec_diff_list)

    rtvec_diff_list_np[:, :3] = rtvec_diff_list_np[:, :3] * 180/PI
    rtvec_diff_list_np[:, 3:] = rtvec_diff_list_np[:, 3:] * norm_factor


    proj_mov_numpy0 = np.array(proj_mov[0,0,:,:].data.cpu())
    target_numpy0 = np.array((target[0,0,:,:]).data.cpu())
    proj_mov_numpy0 = proj_mov_numpy0.reshape((det_size , det_size))
    target_numpy0 = target_numpy0.reshape((det_size , det_size))

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.imshow(proj_init_numpy0)
    ax1.set_title('Initial Proj')

    ax2.imshow(proj_mov_numpy0)
    ax2.set_title('Moving Proj')

    ax3.imshow(target_numpy0)
    ax3.set_title('Target')

    plt.show(block=False)
    plt.pause(1)
    plt.clf()


def save_test_animation(fig, SAVE_PATH, iter, TEST_ID, proj_mov, proj_init_numpy0, target, det_size, norm_factor,\
                   network_sim_list, rtvec_diff_list, rtvec_grad_list, riem_grad_list):
    riem_grad_list_np = np.array(riem_grad_list)
    rtvec_grad_list_np = np.array(rtvec_grad_list)
    network_sim_list_np = np.array(network_sim_list)
    rtvec_diff_list_np = np.array(rtvec_diff_list)

    rtvec_diff_list_np[:, :3] = rtvec_diff_list_np[:, :3] * 180/PI
    rtvec_diff_list_np[:, 3:] = rtvec_diff_list_np[:, 3:] * norm_factor


    proj_mov_numpy0 = np.array(proj_mov[0,0,:,:].data.cpu())
    target_numpy0 = np.array((target[0,0,:,:]).data.cpu())
    proj_mov_numpy0 = proj_mov_numpy0.reshape((det_size , det_size))
    target_numpy0 = target_numpy0.reshape((det_size , det_size))

    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    ax1.imshow(proj_init_numpy0, cmap=plt.get_cmap('viridis'), animated=True)
    ax1.set_title('Initial Proj')

    ax2.imshow(proj_mov_numpy0, cmap=plt.get_cmap('viridis'), animated=True)
    ax2.set_title('Moving Proj')

    ax3.imshow(target_numpy0, cmap=plt.get_cmap('viridis'), animated=True)
    ax3.set_title('Target')

    ax4.plot(rtvec_diff_list_np[:, 0], color='darkred', marker='o')
    ax4.plot(rtvec_diff_list_np[:, 1], color='red', marker='o')
    ax4.plot(rtvec_diff_list_np[:, 2], color='lightsalmon', marker='o')
    ax4.set_title('Rotation Diff')
    ax4.set_ylabel('Rotation/degree')
    ax4.set_xlabel('iteration')
    ax4.tick_params(axis='y', labelcolor='tab:red')
    ax4.legend(['ax', 'ay', 'az'])
    ax4.grid()

    ax5.plot(rtvec_diff_list_np[:, 3], color='navy', marker='^')
    ax5.plot(rtvec_diff_list_np[:, 4], color='blue', marker='^')
    ax5.plot(rtvec_diff_list_np[:, 5], color='deepskyblue', marker='^')
    ax5.set_title('Translation Diff')
    ax5.set_ylabel('Translation/mm')
    ax5.legend(['x', 'y', 'z'])
    ax5.tick_params(axis='y', labelcolor='tab:blue')

    iterfig_dir = SAVE_PATH + '/movie/iterfigs_' + TEST_ID
    if not os.path.exists(iterfig_dir):
        os.mkdir(iterfig_dir)
        print("Folder ", iterfig_dir, " Created")
    else:
        print("Folder ", iterfig_dir, " already exists")

    plt.savefig(iterfig_dir + '/Fig'+str(iter).zfill(4)+'.jpg')

def save_test_animation_comb(fig, SAVE_PATH, iter, TEST_ID, proj_mov, proj_init_numpy0, target, det_size, norm_factor,\
                   network_sim_list, ncc_sim_list, rtvec_diff_list, rtvec_grad_list, riem_grad_list, switch=False):
    riem_grad_list_np = np.array(riem_grad_list)
    rtvec_grad_list_np = np.array(rtvec_grad_list)
    network_sim_list_np = np.array(network_sim_list)
    network_sim_STD = np.std(network_sim_list_np[-10:])
    ncc_sim_list_np = np.array(ncc_sim_list)
    rtvec_diff_list_np = np.array(rtvec_diff_list)

    rtvec_diff_list_np[:, :3] = rtvec_diff_list_np[:, :3] * 180/PI
    rtvec_diff_list_np[:, 3:] = rtvec_diff_list_np[:, 3:] * norm_factor


    proj_mov_numpy0 = np.array(proj_mov[0,0,:,:].data.cpu())
    target_numpy0 = np.array((target[0,0,:,:]).data.cpu())
    proj_mov_numpy0 = proj_mov_numpy0.reshape((det_size , det_size))
    target_numpy0 = target_numpy0.reshape((det_size , det_size))

    ax1 = fig.add_subplot(241)
    ax2 = fig.add_subplot(242)
    ax3 = fig.add_subplot(243)
    ax4 = fig.add_subplot(245)
    ax5 = fig.add_subplot(246)
    ax6 = fig.add_subplot(247)
    ax7 = fig.add_subplot(248)

    ax1.imshow(proj_init_numpy0, cmap=plt.get_cmap('viridis'), animated=True)
    ax1.set_title('Initial Proj')

    ax2.imshow(proj_mov_numpy0, cmap=plt.get_cmap('viridis'), animated=True)
    ax2.set_title('Moving Proj')

    ax3.imshow(target_numpy0, cmap=plt.get_cmap('viridis'), animated=True)
    ax3.set_title('Target')

    ax4.plot(rtvec_diff_list_np[:, 0], color='darkred', marker='o')
    ax4.plot(rtvec_diff_list_np[:, 1], color='red', marker='o')
    ax4.plot(rtvec_diff_list_np[:, 2], color='lightsalmon', marker='o')
    ax4.set_title('Rotation Diff')
    ax4.set_ylabel('Rotation/degree')
    ax4.set_xlabel('iteration')
    ax4.tick_params(axis='y', labelcolor='tab:red')
    ax4.legend(['ax', 'ay', 'az'])
    ax4.grid()

    ax5.plot(rtvec_diff_list_np[:, 3], color='navy', marker='^')
    ax5.plot(rtvec_diff_list_np[:, 4], color='blue', marker='^')
    ax5.plot(rtvec_diff_list_np[:, 5], color='deepskyblue', marker='^')
    ax5.set_title('Translation Diff')
    ax5.set_ylabel('Translation/mm')
    ax5.legend(['x', 'y', 'z'])
    ax5.tick_params(axis='y', labelcolor='tab:blue')

    if switch:
        ax6.plot(ncc_sim_list_np, marker='o', color='red')
        ax6.set_title('NOW: NCC Similarity')
    else:
        ax6.plot(ncc_sim_list_np, marker='o', color='blue')
        ax6.set_title('NCC Similarity')

    ax6.set_xlabel('iteration')
    ax6.grid()

    if switch:
        ax7.plot(network_sim_list_np, marker='o', color='blue')
        ax7.set_title('Network Similarity--STD:{:.4f}'.format(network_sim_STD))
    else:
        ax7.plot(network_sim_list_np, marker='o', color='red')
        ax7.set_title('NOW: Network Similarity--STD:{:.4f}'.format(network_sim_STD))

    ax7.set_xlabel('iteration')
    ax7.grid()

    iterfig_dir = SAVE_PATH + '/movie/iterfigs_' + TEST_ID
    if iter == 0:
        if not os.path.exists(iterfig_dir):
            os.mkdir(iterfig_dir)
            print("Folder ", iterfig_dir, " Created")
        else:
            print("Folder ", iterfig_dir, " already exists")

    plt.savefig(iterfig_dir + '/Fig'+str(iter).zfill(4)+'.jpg')


def save_realtest_animation(fig, SAVE_PATH, iter, TEST_ID, proj_mov, proj_init_numpy0, target, det_size, norm_factor,\
                   network_sim_list, rtvec_diff_list, rtvec_grad_list, riem_grad_list):
    riem_grad_list_np = np.array(riem_grad_list)
    rtvec_grad_list_np = np.array(rtvec_grad_list)
    network_sim_list_np = np.array(network_sim_list)
    rtvec_diff_list_np = np.array(rtvec_diff_list)

    rtvec_diff_list_np[:, :3] = rtvec_diff_list_np[:, :3] * 180/PI
    rtvec_diff_list_np[:, 3:] = rtvec_diff_list_np[:, 3:] * norm_factor


    proj_mov_numpy0 = np.array(proj_mov[0,0,:,:].data.cpu())
    target_numpy0 = np.array((target[0,0,:,:]).data.cpu())
    proj_mov_numpy0 = proj_mov_numpy0.reshape((det_size , det_size))
    target_numpy0 = target_numpy0.reshape((det_size , det_size))

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.imshow(proj_init_numpy0, cmap=plt.get_cmap('viridis'), animated=True)
    ax1.set_title('Initial Proj')

    ax2.imshow(proj_mov_numpy0, cmap=plt.get_cmap('viridis'), animated=True)
    ax2.set_title('Moving Proj')

    ax3.imshow(target_numpy0, cmap=plt.get_cmap('viridis'), animated=True)
    ax3.set_title('Target')

    iterfig_dir = SAVE_PATH + '/movie/iterfigs_' + TEST_ID
    if iter == 0:
        if not os.path.exists(iterfig_dir):
            os.mkdir(iterfig_dir)
            print("Folder ", iterfig_dir, " Created")
        else:
            print("Folder ", iterfig_dir, " already exists")

    plt.savefig(iterfig_dir + '/Fig'+str(iter).zfill(4)+'.jpg')

def save_realtest_animation_comb(fig, SAVE_PATH, iter, TEST_ID, proj_mov, proj_init_numpy0, target, det_size, norm_factor,\
                   network_sim_list, ncc_sim_list, rtvec_diff_list, rtvec_grad_list, riem_grad_list, switch=False):
    riem_grad_list_np = np.array(riem_grad_list)
    rtvec_grad_list_np = np.array(rtvec_grad_list)
    network_sim_list_np = np.array(network_sim_list)
    network_sim_STD = np.std(network_sim_list_np[-10:])
    ncc_sim_list_np = np.array(ncc_sim_list)
    rtvec_diff_list_np = np.array(rtvec_diff_list)

    rtvec_diff_list_np[:, :3] = rtvec_diff_list_np[:, :3] * 180/PI
    rtvec_diff_list_np[:, 3:] = rtvec_diff_list_np[:, 3:] * norm_factor


    proj_mov_numpy0 = np.array(proj_mov[0,0,:,:].data.cpu())
    target_numpy0 = np.array((target[0,0,:,:]).data.cpu())
    proj_mov_numpy0 = proj_mov_numpy0.reshape((det_size , det_size))
    target_numpy0 = target_numpy0.reshape((det_size , det_size))

    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)

    ax1.imshow(proj_init_numpy0, cmap=plt.get_cmap('viridis'), animated=True)
    ax1.set_title('Initial Proj')

    ax2.imshow(proj_mov_numpy0, cmap=plt.get_cmap('viridis'), animated=True)
    ax2.set_title('Moving Proj')

    ax3.imshow(target_numpy0, cmap=plt.get_cmap('viridis'), animated=True)
    ax3.set_title('Target')

    if switch:
        ax4.plot(ncc_sim_list_np, marker='o', color='red')
        ax4.set_title('NOW: NCC Similarity')
    else:
        ax4.plot(ncc_sim_list_np, marker='o', color='blue')
        ax4.set_title('NCC Similarity')

    ax4.set_xlabel('iteration')
    ax4.grid()

    if switch:
        ax5.plot(network_sim_list_np, marker='o', color='blue')
        ax5.set_title('Network Similarity--STD:{:.4f}'.format(network_sim_STD))
    else:
        ax5.plot(network_sim_list_np, marker='o', color='red')
        ax5.set_title('NOW: Network Similarity--STD:{:.4f}'.format(network_sim_STD))

    ax5.set_xlabel('iteration')
    ax5.grid()

    iterfig_dir = SAVE_PATH + '/movie/iterfigs_' + TEST_ID
    if iter == 0:
        if not os.path.exists(iterfig_dir):
            os.mkdir(iterfig_dir)
            print("Folder ", iterfig_dir, " Created")
        else:
            print("Folder ", iterfig_dir, " already exists")

    plt.savefig(iterfig_dir + '/Fig'+str(iter).zfill(4)+'.jpg')


def save_stattest_iter(iterfig_dir, iter, two_fold, proj_mov, proj_init_numpy0, target, det_size, norm_factor,\
                   network_sim_list, ncc_sim_list, rtvec_diff_list, rtvec_grad_list, riem_grad_list, switch=False):
    fig = plt.figure(figsize=(15, 9))
    network_sim_list_np = np.array(network_sim_list)
    network_sim_STD = np.std(network_sim_list_np[-10:])
    ncc_sim_list_np = np.array(ncc_sim_list)
    ncc_sim_STD = np.std(ncc_sim_list_np[-10:])
    rtvec_diff_list_np = np.array(rtvec_diff_list)

    rtvec_diff_list_np[:, :3] = rtvec_diff_list_np[:, :3] * 180/PI
    rtvec_diff_list_np[:, 3:] = rtvec_diff_list_np[:, 3:] * norm_factor


    proj_mov_numpy0 = np.array(proj_mov[0,0,:,:].data.cpu())
    target_numpy0 = np.array((target[0,0,:,:]).data.cpu())
    proj_mov_numpy0 = proj_mov_numpy0.reshape((det_size , det_size))
    #proj_mov_grad = np.gradient(proj_mov_numpy0, edge_order=1)
    target_numpy0 = target_numpy0.reshape((det_size , det_size))
    #target_numpy0 = target_numpy0 + proj_mov_grad[0]*2

    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    ax1.imshow(proj_init_numpy0)
    ax1.set_title('Initial Proj')

    ax2.imshow(proj_mov_numpy0)
    ax2.set_title('Moving Proj')

    ax3.imshow(target_numpy0)
    ax3.set_title('Target')

    ax4.plot(rtvec_diff_list_np[:, 0], color='darkred', marker='o')
    ax4.plot(rtvec_diff_list_np[:, 1], color='red', marker='o')
    ax4.plot(rtvec_diff_list_np[:, 2], color='lightsalmon', marker='o')
    ax4.set_title('Rvec Diff')
    ax4.set_ylabel('degree')
    ax4.set_xlabel('iteration')
    ax4.tick_params(axis='y', labelcolor='tab:red')
    ax4.legend(['ax', 'ay', 'az'])
    ax4.grid()

    ax42 = ax4.twinx()
    ax42.plot(rtvec_diff_list_np[:, 3], color='navy', marker='^')
    ax42.plot(rtvec_diff_list_np[:, 4], color='blue', marker='^')
    ax42.plot(rtvec_diff_list_np[:, 5], color='deepskyblue', marker='^')
    ax42.set_ylabel('mm')
    ax42.legend(['x', 'y', 'z'])
    ax42.tick_params(axis='y', labelcolor='tab:blue')

    if switch:
        ax5.plot(ncc_sim_list_np, marker='o', color='red')
        ax5.set_title('NOW: Grad-NCC Similarity--STD:{:.4f}'.format(ncc_sim_STD))
    else:
        ax5.plot(ncc_sim_list_np, marker='o', color='blue')
        ax5.set_title('Grad-NCC Similarity--STD:{:.4f}'.format(ncc_sim_STD))

    ax5.set_xlabel('iteration')
    ax5.grid()

    if switch:
        ax6.plot(network_sim_list_np, marker='o', color='blue')
        ax6.set_title('Network Similarity--STD:{:.4f}'.format(network_sim_STD))
    else:
        ax6.plot(network_sim_list_np, marker='o', color='red')
        ax6.set_title('NOW: Network Similarity--STD:{:.4f}'.format(network_sim_STD))

    ax6.set_xlabel('iteration')
    ax6.grid()

    plt.savefig(iterfig_dir + '/Fig'+str(iter).zfill(4)+'_'+str(two_fold)+'.jpg')


def save_vali_iter(SAVE_PATH, epoch, idx, proj_mov, proj_init_numpy0, target, det_size, norm_factor,\
               network_sim_list, rtvec_diff_list, rtvec_grad_diff_list):
    fig = plt.figure(figsize=(15, 9))
    rtvec_grad_diff_list_np = np.array(rtvec_grad_diff_list)
    network_sim_list_np = np.array(network_sim_list)
    network_sim_STD = np.std(network_sim_list_np[-10:])
    rtvec_diff_list_np = np.array(rtvec_diff_list)

    rtvec_diff_list_np[:, :3] = rtvec_diff_list_np[:, :3] * 180/PI
    rtvec_diff_list_np[:, 3:] = rtvec_diff_list_np[:, 3:] * norm_factor


    proj_mov_numpy0 = np.array(proj_mov[0,0,:,:].data.cpu())
    target_numpy0 = np.array((target[0,0,:,:]).data.cpu())
    proj_mov_numpy0 = proj_mov_numpy0.reshape((det_size , det_size))
    target_numpy0 = target_numpy0.reshape((det_size , det_size))

    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    ax1.imshow(proj_init_numpy0, cmap=plt.get_cmap('viridis'), animated=True)
    ax1.set_title('Initial Proj')

    ax2.imshow(proj_mov_numpy0, cmap=plt.get_cmap('viridis'), animated=True)
    ax2.set_title('Moving Proj')

    ax3.imshow(target_numpy0, cmap=plt.get_cmap('viridis'), animated=True)
    ax3.set_title('Target')

    ax4.plot(rtvec_diff_list_np[:, 0], color='darkred', marker='o')
    ax4.plot(rtvec_diff_list_np[:, 1], color='red', marker='o')
    ax4.plot(rtvec_diff_list_np[:, 2], color='lightsalmon', marker='o')
    ax4.set_title('Rvec Diff')
    ax4.set_ylabel('degree')
    ax4.set_xlabel('iteration')
    ax4.tick_params(axis='y', labelcolor='tab:red')
    ax4.legend(['ax', 'ay', 'az'])
    ax4.grid()

    ax42 = ax4.twinx()
    ax42.plot(rtvec_diff_list_np[:, 3], color='navy', marker='^')
    ax42.plot(rtvec_diff_list_np[:, 4], color='blue', marker='^')
    ax42.plot(rtvec_diff_list_np[:, 5], color='deepskyblue', marker='^')
    ax42.set_ylabel('mm')
    ax42.legend(['x', 'y', 'z'])
    ax42.tick_params(axis='y', labelcolor='tab:blue')

    ax5.plot(rtvec_grad_diff_list_np[:, 0], color='darkred', marker='o')
    ax5.plot(rtvec_grad_diff_list_np[:, 1], color='red', marker='o')
    ax5.plot(rtvec_grad_diff_list_np[:, 2], color='lightsalmon', marker='o')
    ax5.set_title('Rvec Diff')
    ax5.set_xlabel('iteration')
    ax5.tick_params(axis='y', labelcolor='tab:red')
    ax5.legend(['ax', 'ay', 'az'])
    ax5.grid()

    ax52 = ax5.twinx()
    ax52.plot(rtvec_grad_diff_list_np[:, 3], color='navy', marker='^')
    ax52.plot(rtvec_grad_diff_list_np[:, 4], color='blue', marker='^')
    ax52.plot(rtvec_grad_diff_list_np[:, 5], color='deepskyblue', marker='^')
    ax52.legend(['x', 'y', 'z'])
    ax52.tick_params(axis='y', labelcolor='tab:blue')

    ax6.plot(network_sim_list_np, marker='o', color='blue')
    ax6.set_title('Network Similarity--STD:{:.4f}'.format(network_sim_STD))

    vali_epoch_dir = SAVE_PATH + '/stat_figs/vali_epoch'+str(epoch)
    if not os.path.exists(vali_epoch_dir):
        os.mkdir(vali_epoch_dir)
        print("Folder ", vali_epoch_dir, " Created")
    else:
        print("Folder ", vali_epoch_dir, " already exists")
    vali_epoch_name = vali_epoch_dir + '/Fig_idx' + str(idx) + '.jpg'
    plt.savefig(vali_epoch_name)
    plt.close(fig)


'''
plt.imshow(target.detach().cpu().numpy()[0,0,:,:].reshape(det_size, det_size))
plt.title('0')
plt.show()
plt.imshow(target.detach().cpu().numpy()[1,0,:,:].reshape(det_size, det_size))
plt.title('1')
plt.show()
plt.imshow(target.detach().cpu().numpy()[2,0,:,:].reshape(det_size, det_size))
plt.title('2')
plt.show()
'''
