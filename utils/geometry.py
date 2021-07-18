import torch
import numpy as np
from numba import njit, prange
from pyquaternion import Quaternion


@njit
def vox2world(vol_origin, vox_coords, vox_size):
    """
    Converts voxel grid coordinates to world coordinates

    :param vol_origin: Origin of the volume in world coordinates.
    :parma vol_coords: List of all grid coordinates in the volume.
    :param vol_size: Size of volume.
    :retrun: Grid points under world coordinates.
    """

    vol_origin = vol_origin.astype(np.float32)
    vox_coords = vox_coords.astype(np.float32)
    cam_pts = torch.empty_like(vox_coords, dtype=torch.float32)

    for i in prange(vox_coords.shape[0]):
        cam_pts[i, :] = vol_origin + (vox_size * vox_coords[i, :])

    return cam_pts


@njit
def cam2pix(cam_pts, intrinsics):
    """
    Convert points in camera coordinate to pixel coordinates

    :param cam_pts: Points in camera coordinates.
    :param intrinsics: Vamera intrinsics.
    :return: Pixel coordinate (u,v) cooresponding to input points.
    """
    intrinsics = intrinsics.astype(np.float32)
    fx = intrinsics[0,0]
    fy = intrinsics[1,1]
    cx = intrinsics[0,2]
    cy = intrinsics[1,2]
    
    pix = torch.empty((cam_pts.shape[0], 2), dtype=torch.int64)
    for i in prange(cam_pts.shape[0]):
        pix[i, 0] = int(np.round(cam_pts[i,0] * fx / cam_pts[i, 2]) + cx)
        pix[i, 1] = int(np.round(cam_pts[i,1] * fy / cam_pts[i, 2]) + cy)

    return pix


@njit
def ridgid_transform(points, transform):
    """
    Apply rigid transform (4,4) on points

    :param points: Points, shape (N,3).
    :param transform: Tranform matrix, shape (4,4).
    :return: Points after transform.
    """
    transform = torch.from_numpy(transform, device=points.device)
    points_h = torch.cat([points, torch.ones((points.shape[0], 1))], 1) 
    points_h = points_h @ transform
    
    return points_h[:, :3]