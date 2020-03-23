import torch
import ProSTGrid
import torch.nn as nn
import numpy as np
from util import _bilinear_interpolate_no_torch_5D
import torchgeometry as tgm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from posevec2mat import pose_vec2mat, inv_pose_vec, raydist_range

device = torch.device("cuda")

_BOTTLENECK_EXPANSION = 4

class Pelvis_Dataset(Dataset):
    """FBG-dataDriven-dataSet"""

    def __init__(self, theta_npy, proj_path, det_size, train=True, transform=None):

        self.theta_data = np.load(theta_npy)
        self.trial_num = self.theta_data.shape[0]
        self.proj_path = proj_path
        self.det_size = det_size

        if train:
            self.train_size = self.trial_num
        else:
            self.vali_size = self.trial_num

        self.transform = transform
        self.train_flag = train

    def __len__(self):
        if self.train_flag:
            return self.train_size
        else:
            return self.vali_size

    def __getitem__(self, idx):
        theta = np.array(self.theta_data[idx, :]).astype('float')
        proj = Image.open(self.proj_path + "/proj_" + str(idx).zfill(3) + ".tiff")
        # proj = proj.resize((self.det_size, self.det_size), Image.BILINEAR)
        # proj = (proj -  np.min(proj))/(np.max(proj)-np.min(proj))
        proj = np.expand_dims(np.array(proj), axis=0)
        sample = {'img': proj, 'theta': theta}

        if self.transform:
            sample = self.transform(sample)

        sample['img'].astype(float)
        sample['theta'].astype(float)
        return sample

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ProST(nn.Module):
    def __init__(self, param):
        super(ProST, self).__init__()
        self.src = param[0]
        self.det = param[1]
        self.pix_spacing = param[2]
        self.step_size = param[3]

    def forward(self, x, y, rtvec, corner_pt):
        BATCH_SIZE = rtvec.size()[0]
        H = y.size()[2]
        W = y.size()[3]

        transform_mat4x4 = tgm.rtvec_to_pose(rtvec)
        transform_mat3x4 = transform_mat4x4[:, :3, :]
        dist_min, dist_max = raydist_range(transform_mat3x4, corner_pt, self.src)

        grid = ProSTGrid.forward(corner_pt, y.size(), dist_min.data, dist_max.data,\
                                     self.src, self.det, self.pix_spacing, self.step_size, False)

        grid_trans = grid.bmm(transform_mat3x4.transpose(1,2)).view(BATCH_SIZE, H, W, -1, 3)

        x_3d_ad = _bilinear_interpolate_no_torch_5D(x, grid_trans)
        x_2d_ad = torch.sum(x_3d_ad, dim=-1)

        return x_2d_ad

class ProST_init(nn.Module):
    def __init__(self, param):
        super(ProST_init, self).__init__()
        self.src = param[0]
        self.det = param[1]
        self.pix_spacing = param[2]
        self.step_size = param[3]

    def forward(self, x, y, transform_mat3x4, corner_pt):
        BATCH_SIZE = transform_mat3x4.size()[0]
        H = y.size()[2]
        W = y.size()[3]

        dist_min, dist_max = raydist_range(transform_mat3x4, corner_pt, self.src)

        grid = ProSTGrid.forward(corner_pt, y.size(), dist_min.data, dist_max.data, self.src, self.det, self.pix_spacing, self.step_size, False)
        grid_trans = grid.bmm(transform_mat3x4.transpose(1,2)).view(BATCH_SIZE, H, W, -1, 3)
        x = _bilinear_interpolate_no_torch_5D(x, grid_trans)
        x = torch.sum(x, dim=-1)

        return x

class RegiNet(nn.Module):
    def __init__(self, param, det_size):
        super(RegiNet, self).__init__()
        self.src = param[0]
        self.det = param[1]
        self.pix_spacing = param[2]
        self.step_size = param[3]

        self._3D_conv = nn.Sequential(
            nn.Conv3d(1, 4, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(4, 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(16, 8, 3, 1, 1),
            nn.ReLU(),
            nn.Conv3d(8, 3, 3, 1, 1),
            nn.ReLU()
        )

        self._2Dconv_encode_x = nn.Sequential(
                                           nn.Conv2d(3, 16, 3, 1, 1),
                                           nn.ReLU(),
                                           Bottleneck(16, 16),
                                           nn.MaxPool2d(2),
                                           nn.Conv2d(16, 64, 3, 1, 1),
                                           nn.ReLU(),
                                           Bottleneck(64, 64),
                                           nn.MaxPool2d(2),
                                           nn.Conv2d(64, 128, 3, 1, 1),
                                           nn.ReLU(),
                                           Bottleneck(128, 128),
                                           nn.MaxPool2d(2),
                                           nn.Conv2d(128, 64, 3, 1, 1),
                                           nn.ReLU()
                                           )

        self._2Dconv_encode_y = nn.Sequential(
                                           nn.Conv2d(1, 16, 3, 1, 1),
                                           nn.ReLU(),
                                           Bottleneck(16, 16),
                                           nn.MaxPool2d(2),
                                           nn.Conv2d(16, 64, 3, 1, 1),
                                           nn.ReLU(),
                                           Bottleneck(64, 64),
                                           nn.MaxPool2d(2),
                                           nn.Conv2d(64, 128, 3, 1, 1),
                                           nn.ReLU(),
                                           Bottleneck(128, 128),
                                           nn.MaxPool2d(2),
                                           nn.Conv2d(128, 64, 3, 1, 1),
                                           nn.ReLU()
                                           )

    def forward(self, x, y, theta, corner_pt):
        x_exp = x.repeat(1,3,1,1,1)
        x_3d = x_exp + self._3D_conv(x)
        BATCH_SIZE = theta.size()[0]
        H = y.size()[2]
        W = y.size()[3]

        transform_mat4x4 = tgm.rtvec_to_pose(theta)
        transform_mat3x4 = transform_mat4x4[:, :3, :]
        dist_min, dist_max = raydist_range(transform_mat3x4, corner_pt, self.src)

        grid = ProSTGrid.forward(corner_pt, y.size(), dist_min.data, dist_max.data,\
                                     self.src, self.det, self.pix_spacing, self.step_size, False)
        grid_trans = grid.bmm(transform_mat3x4.transpose(1,2)).view(BATCH_SIZE, H, W, -1, 3)
        x_3d = _bilinear_interpolate_no_torch_5D(x_3d, grid_trans)
        x_2d = torch.sum(x_3d, dim=-1)

        x_3d_ad = _bilinear_interpolate_no_torch_5D(x, grid_trans)
        x_2d_ad = torch.sum(x_3d_ad, dim=-1)

        y_out = self._2Dconv_encode_y(y)
        x_out = self._2Dconv_encode_x(x_2d)

        return x_out, y_out, x_2d_ad
