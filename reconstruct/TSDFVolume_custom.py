import threading
import mcubes
import torch
import numpy as np
import open3d as o3d

from utils import geometry



class TSDFVolume:
    """
    Volumetric with TSDF representation
    """

    def __init__(self, vol_bounds: np.ndarray, vol_size: float, use_gpu: bool=True) -> None:
        """
        Constructor

        :param vol_bounds: An ndarray is shape (3,2), define the min & max bounds of voxels.
        :param voxel_size: Voxel size in meters.
        :param use_gpu: Use GPU for voxel update.
        """

        assert vol_bounds.shape == (3,2), "vol_bounds should be of shape (3,2)"

        # Volumetric parameters
        self._vol_bounds = vol_bounds
        self._vox_size = vol_size
        self._trunc_margin = 5 * self._vol_size    # truncation on SDF
        
        self._vol_dim = np.ceil((self._vol_bounds[:, 1] - self._vol_bounds[:, 2]) / self._vox_size).copy(order='C').astype(int)
        self._vol_bounds[:,1] = self._vol_bounds[:, 0] + self._vol_dim * self._vox_size
        
        # coordinate origin of the volume, where to place the volume ?
        # TODO: manually set the origin ?
        self._vol_origin = self._vol_bounds[:, 0].copy(order='C').astype(np.float32)    

        # Assign Volumetric
        self._use_gpu = use_gpu
        if self._use_gpu:
            if torch.cuda.is_available():
                self._divice = torch.device("cude:0")
            else:
                print("Not available CUDA device, using CPU mode")
                self._divice = torch.device("cpu")
        else:
            self._divice = torch.device("cpu")

        self._tsdf_vol = torch.ones(shape=self._vol_dim, device=self._divice, dtype=torch.float32)
        self._weight_vol = torch.zeros(shape=self._vol_dim, device=self._divice, dtype=torch.float32)

        print("Create TSDF Volume with size {} x {} x {}".format(self._vol_dim[0], self._vol_dim[1], self._vol_dim[2]))

        xx, yy, zz = torch.meshgrid(
            torch.arange(self._vol_dim[0]),
            torch.arange(self._vol_dim[1]),
            torch.arange(self._vol_dim[2])
        )

        self._vox_coords = torch.cat([
            xx.reshape(1, -1),
            yy.reshape(1, -1),
            zz.reshape(1, -1)
        ], dim=0).int().T
        
        if self._use_gpu:
            self._vox_coords.cuda()

        # Mesh paramters
        # self._mesh_vertices_cache
        # self._mesh_triangles_cache
        self._mesh = o3d.geometry.TriangleMesh()
        

    def integrate(self, depth_img, intrinsic, cam_pose, weights: float=1.):
        """
        Integrate an depth image to the TSDF volume

        :param depth_img: depth image with depth value in meter.
        :param intrinsics: camera intrinsics of shape (3,3).
        :param cam_pose: camera pose, transform matrix of shape (4,4)
        :param weights: weights assign for current observation, higher value indicate higher confidence ?
        """

        img_h, img_w = depth_img.shape

        camera_points = geometry.vox2world(self._vol_origin, self._vox_coords, self._vox_size)
        camera_points = geometry.ridgid_transform(camera_points, np.linalg.inv(cam_pose))
        
        pixels = geometry.cam2pix(camera_points, intrinsic)
        pixels_x, pixels_y = pixels[:, 0], pixels[:, 1]
        pixels_z = camera_points[:, 2]