import threading
import mcubes
import torch
import time
import os, sys
import numpy as np
import open3d as o3d
from numba import njit, prange

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import geometry


class TSDFVolume:
    """
    Volumetric with TSDF representation
    """

    def __init__(self, vol_bounds: np.ndarray, vox_size: float, use_gpu: bool=True, verbose: bool=True) -> None:
        """
        Constructor

        :param vol_bounds: An ndarray is shape (3,2), define the min & max bounds of voxels.
        :param voxel_size: Voxel size in meters.
        :param use_gpu: Use GPU for voxel update.
        :param verbose: Print verbose message or not.
        """

        vol_bounds = np.asarray(vol_bounds)
        assert vol_bounds.shape == (3,2), "vol_bounds should be of shape (3,2)"

        self._verbose = verbose
        self._use_gpu = use_gpu

        # Volumetric parameters
        self._vol_bounds = vol_bounds
        self._vox_size = vox_size
        self._trunc_margin = 5 * self._vox_size    # truncation on SDF
        
        self._vol_dim = np.ceil((self._vol_bounds[:, 1] - self._vol_bounds[:, 0]) / self._vox_size).copy(order='C').astype(int)
        self._vol_bounds[:,1] = self._vol_bounds[:, 0] + self._vol_dim * self._vox_size
        self._vol_dim = tuple(self._vol_dim)

        if self._verbose:
            print("# Create TSDF Volume with size {} x {} x {}".format(self._vol_dim[0], self._vol_dim[1], self._vol_dim[2])) 

        # Check GPU
        if self._use_gpu:
            if torch.cuda.is_available():
                if self._verbose:
                    print("# Using GPU mode")
                self._device = torch.device("cuda:0")
            else:
                if self._verbose:
                    print("# Not available CUDA device, using CPU mode")
                self._device = torch.device("cpu")
        else:
            if self._verbose:
                print("# Using CPU mode")
            self._device = torch.device("cpu")

        # Coordinate origin of the volume, set as the min value of volume bounds
        self._vol_origin = torch.tensor(self._vol_bounds[:, 0].copy(order='C'), device=self._device).float()

        # Grid coordinates of voxels
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
            self._vox_coords = self._vox_coords.cuda()

        # World coordinates of voxel centers
        self._world_coords = geometry.vox2world(self._vol_origin, self._vox_coords, self._vox_size)

        # TSDF & weights
        self._tsdf_vol = torch.ones(size=self._vol_dim, device=self._device, dtype=torch.float32)
        self._weight_vol = torch.zeros(size=self._vol_dim, device=self._device, dtype=torch.float32)

        # Mesh paramters
        self._mesh = o3d.geometry.TriangleMesh()
        

    def get_volume(self) -> torch.Tensor:
        """
        Get TSDF volume.
        """
        return self._tsdf_vol


    def get_mesh(self):
        """
        Get mesh.
        """
        return self._mesh


    def integrate(self, depth_img, intrinsic, cam_pose, weight: float=1.):
        """
        Integrate an depth image to the TSDF volume

        :param depth_img: depth image with depth value in meter.
        :param intrinsics: camera intrinsics of shape (3,3).
        :param cam_pose: camera pose, transform matrix of shape (4,4)
        :param weight: weight assign for current frame, higher value indicate higher confidence
        """

        time_begin = time.time()

        img_h, img_w = depth_img.shape
        depth_img = torch.tensor(depth_img, device=self._device).float()
        cam_pose = torch.tensor(cam_pose, device=self._device).float()
        intrinsic = torch.tensor(intrinsic, device=self._device).float()

        # TODO: 
        # Better way to select valid voxels.
        # - Current: 
        #   -> Back project all voxels to frame pixels according to current camera pose. 
        #   -> Select valid pixels within frame size.
        # - Possible: 
        #   -> Project pixel to voxel coordinates 
        #   -> hash voxel coordinates 
        #   -> dynamically allocate voxel chunks

        # Get the world coordinates of all voxels
        # world_points = geometry.vox2world(self._vol_origin, self._vox_coords, self._vox_size)

        # Get voxel centers under camera coordinates
        world_points = geometry.ridgid_transform(self._world_coords, cam_pose.inverse())
        
        # Get the pixel coordinates (u,v) of all voxels under current camere pose
        # Multiple voxels can be projected to a same (u,v)
        voxel_uv = geometry.cam2pix(world_points, intrinsic)
        voxel_u, voxel_v = voxel_uv[:, 0], voxel_uv[:, 1]
        voxel_z = world_points[:, 2]

        # Filter out voxels points that visible in current frame
        pixel_mask = torch.logical_and(voxel_u >= 0,
                     torch.logical_and(voxel_u < img_w,
                     torch.logical_and(voxel_v >= 0,
                     torch.logical_and(voxel_v < img_h,
                     voxel_z > 0))))
        
        # Get depth value
        depth_value = torch.zeros(voxel_u.shape, device=self._device)
        depth_value[pixel_mask] = depth_img[voxel_v[pixel_mask].long(), voxel_u[pixel_mask].long()]

        # Compute and Integrate TSDF
        sdf_value = depth_value - voxel_z         # Compute SDF
        voxel_mask = torch.logical_and(depth_value > 0, sdf_value >= -self._trunc_margin)        # Truncate SDF
        tsdf_value = sdf_value[voxel_mask]
        tsdf_value = torch.minimum(torch.ones_like(sdf_value, device=self._device), sdf_value / self._trunc_margin)

        # Get coordinates of valid voxels with valid TSDF value
        valid_vox_x = self._vox_coords[voxel_mask, 0].long()
        valid_vox_y = self._vox_coords[voxel_mask, 1].long()
        valid_vox_z = self._vox_coords[voxel_mask, 2].long()

        # Update TSDF of cooresponding voxels
        weight_old = self._weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]
        tsdf_old = self._tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]

        tsdf_new, weight_new = self.update_tsdf(tsdf_old, tsdf_value, weight_old, weight)

        self._tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_new
        self._weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = weight_new

        if self._verbose:
            print("# Update {} voxels.".format(len(tsdf_new)))
            print("# Integration Timing: {:.5f} (second).".format(time.time() - time_begin))


    def extract_mesh(self):
        """
        Extract mesh from current TSDF volume.
        """

        time_begin = time.time()

        if self._use_gpu:
            tsdf_vol = self._tsdf_vol.cpu().numpy()
            vol_origin = self._vol_origin.cpu().numpy()
        else:
            tsdf_vol = self._tsdf_vol.numpy()
            vol_origin = self._vol_origin.numpy()

        vertices, triangles = mcubes.marching_cubes(tsdf_vol, 0)
        # Convert vertices in grid coordinates to world coordinates
        vertices = vertices * self._vox_size + vol_origin

        self._mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(float))
        self._mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
        self._mesh.compute_vertex_normals()

        if self._verbose:
            print("# Extracting Mesh: {} Vertices".format(vertices.shape[0]))
            print("# Meshing Timing: {:.5f} (second).".format(time.time() - time_begin))


    def save_mesh(self, filename):
        """
        Save the mesh to .ply file.

        :param filename: Filename of the target .ply file.
        """

        vertices = np.asarray(self._mesh.vertices)
        normals = np.asarray(self._mesh.vertex_normals)

        # Write header
        ply_file = open(filename,'w')
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write("element vertex %d\n"%(vertices.shape[0]))
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property float nx\n")
        ply_file.write("property float ny\n")
        ply_file.write("property float nz\n")
        ply_file.write("end_header\n")

        # Write vertex list
        for i in range(vertices.shape[0]):
            ply_file.write("%f %f %f %f %f %f".format(
                vertices[i,0], vertices[i,1], vertices[i,2],
                normals[i,0], normals[i,1], normals[i,2]
            ))

        ply_file.close()

        if self._verbose:
            print("Saving mesh to {}.".format(filename))



    def update_tsdf(self, tsdf_old, tsdf_new, weight_old, obs_weight):
        """
        Update the TSDF value of given voxel
        V = (wv + WV) / w + W

        :param tsdf_old: Old TSDF values.
        :param tsdf_new: New TSDF values.
        :param weight_old: Voxels weights. 
        :param obs_weight: Weight of current update.
        :return: Updated TSDF values & Updated weights.
        """

        tsdf_vol_int = torch.empty_like(tsdf_old, dtype=torch.float32, device=self._device)
        weight_new = torch.empty_like(weight_old, dtype=torch.float32, device=self._device)

        weight_new = weight_old + obs_weight
        tsdf_vol_int = (weight_old * tsdf_old + obs_weight * tsdf_new) / weight_new

        return tsdf_vol_int, weight_new
