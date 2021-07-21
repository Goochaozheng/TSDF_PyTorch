import torch
import numpy as np
from numba import jit, njit, prange
from pyquaternion import Quaternion


def vox2world(vol_origin: torch.Tensor, vox_coords: torch.Tensor, vox_size) -> torch.Tensor:
    """
    Converts voxel grid coordinates to world coordinates

    :param vol_origin: Origin of the volume in world coordinates, (3,1).
    :parma vol_coords: List of all grid coordinates in the volume, (N,3).
    :param vol_size: Size of volume.
    :retrun: Grid points under world coordinates. Tensor with shape (N, 3)
    """

    cam_pts = torch.empty_like(vox_coords, dtype=torch.float32)
    cam_pts = vol_origin + (vox_size * vox_coords)

    return cam_pts


def cam2pix(cam_pts: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """
    Convert points in camera coordinate to pixel coordinates

    :param cam_pts: Points in camera coordinates, (N,3).
    :param intrinsics: Vamera intrinsics, (3,3).
    :return: Pixel coordinate (u,v) cooresponding to input points. Tensor with shape (N, 2).
    """

    cam_pts_z = cam_pts[:, 2].repeat(3, 1).T
    pix = torch.round((cam_pts @ intrinsics.T ) / cam_pts_z)

    return pix


def ridgid_transform(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """
    Apply rigid transform (4,4) on points

    :param points: Points, shape (N,3).
    :param transform: Tranform matrix, shape (4,4).
    :return: Points after transform.
    """

    points_h = torch.cat([points, torch.ones((points.shape[0], 1), device=points.device)], 1) 
    points_h = (transform @ points_h.T).T
    
    return points_h[:, :3]