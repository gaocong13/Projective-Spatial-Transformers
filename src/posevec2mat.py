import torch
import torch.nn.functional as F

def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:,0], angle[:,1], angle[:,2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, 3:].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:,:3]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat

def inv_pose_vec(transform_mat, pt):
    #  pt: B * 8 * 3 (X, Y, Z)
    # vec: B * 6 (Rx, Ry, Rz, Tx, Ty, Tz)
    pt = pt - transform_mat[:, :3, 3].unsqueeze(1).repeat(1,8,1)
    rot_Mat = transform_mat[:, :3, :3]
    inv_rotMat = torch.inverse(rot_Mat)
    inv_pt = pt.bmm(inv_rotMat)

    return inv_pt

def raydist_range(transform_mat, pt, src):
    inv_pt = inv_pose_vec(transform_mat, pt)
    #inv_pt = inv_pt.squeeze(0)
    inv_pt[:,:,2] = src - inv_pt[:,:,2]
    inv_pt = inv_pt.view(-1, 3)
    dist_pt = torch.sqrt(torch.mul(inv_pt[:,0], inv_pt[:,0]) + torch.mul(inv_pt[:,1], inv_pt[:,1]) + torch.mul(inv_pt[:,2], inv_pt[:,2]))
    dist_min = torch.min(dist_pt)
    dist_max = torch.max(dist_pt)
    return dist_min, dist_max

'''
# Former Definition

def inv_pose_vec(vec, transform_mat, pt):
    #  pt: B * 8 * 3 (X, Y, Z)
    # vec: B * 6 (Rx, Ry, Rz, Tx, Ty, Tz)
    pt = pt - vec[:, 3:].unsqueeze(1).repeat(1,8,1)
    rot_Mat = transform_mat[:, :3, :3]
    inv_rotMat = torch.inverse(rot_Mat)
    inv_pt = pt.bmm(inv_rotMat)

    return inv_pt

def raydist_range(vec, transform_mat, pt, src):
    inv_pt = inv_pose_vec(vec, transform_mat, pt)
    #inv_pt = inv_pt.squeeze(0)
    inv_pt[:,:,2] = src - inv_pt[:,:,2]
    inv_pt = inv_pt.view(-1, 3)
    dist_pt = torch.sqrt(torch.mul(inv_pt[:,0], inv_pt[:,0]) + torch.mul(inv_pt[:,1], inv_pt[:,1]) + torch.mul(inv_pt[:,2], inv_pt[:,2]))
    dist_min = torch.min(dist_pt)
    dist_max = torch.max(dist_pt)
    return dist_min, dist_max
'''
