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

    # vol_origin = vol_origin.astype(np.float32)
    # vox_coords = vox_coords.astype(np.float32)

    # for i in range(vox_coords.shape[0]):
    #     cam_pts[i, :] = vol_origin + (vox_size * vox_coords[i, :])

    # if not hasattr(vol_origin, "device"):
    #     vol_origin = torch.from_numpy(vol_origin)
    # if not hasattr(vox_coords, "device"):
    #     vox_coords = torch.from_numpy(vox_coords)

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
    # intrinsics = intrinsics.astype(np.float32)
    # fx = intrinsics[0,0]
    # fy = intrinsics[1,1]
    # cx = intrinsics[0,2]
    # cy = intrinsics[1,2]
    
    # pix = torch.empty((cam_pts.shape[0], 2), dtype=torch.int64, device=cam_pts.device)
    # for i in range(cam_pts.shape[0]):
    #     pix[i, 0] = int(np.round(cam_pts[i,0] * fx / cam_pts[i, 2]) + cx)
    #     pix[i, 1] = int(np.round(cam_pts[i,1] * fy / cam_pts[i, 2]) + cy)
    
    # TODO: Intrinsic project using torch matrix operation.

    # if not hasattr(cam_pts, "device"):
    #     cam_pts = torch.from_numpy(cam_pts)
    # if not hasattr(intrinsics, "device"):
    #     intrinsics = torch.from_numpy(intrinsics)

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

    # if not hasattr(points, "device"):
    #     points = torch.from_numpy(points)
    # if not hasattr(transform, "device"):
    #     transform = torch.from_numpy(transform)

    points_h = torch.cat([points, torch.ones((points.shape[0], 1), device=points.device)], 1) 
    points_h = (transform @ points_h.T).T
    
    return points_h[:, :3]