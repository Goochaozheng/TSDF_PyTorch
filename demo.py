import time
import cv2
import numpy as np
import argparse
import open3d as o3d

from reconstruct import TSDFVolume
from utils import visualize


vis_param = argparse.Namespace()
vis_param.n_steps_left = 0
vis_param.index = 0
vis_param.n_imgs = 1000
vis_param.stop = True
vis_param.current_pose = np.eye(4)


# def keyStep(vis):
# 	drawNextFrame(vis)
# 	vis_param.n_steps_left += 1
# 	return False


def updateGeometry(geom, name, vis):
	"""
	Update geometry in open3D visualization
	"""

	if name in vis_param.__dict__.keys():
		vis.remove_geometry(geom, reset_bounding_box=False)

	vis.add_geometry(geom, reset_bounding_box=False)
	vis_param.__dict__[name] = geom


def drawNextFrame(vis):
	"""
	Update the TSDF of next frame
	"""

	if vis_param.index >= vis_param.n_imgs or vis_param.n_steps_left <= 0 :
		return False

	print("Fusing frame {}/{}".format(vis_param.index + 1, vis_param.n_imgs))

	# Read RGB-D image and camera pose
	color_image = cv2.cvtColor(cv2.imread("data/demo/frame-%06d.color.jpg" % (vis_param.index)), cv2.COLOR_BGR2RGB)
	depth_im = cv2.imread("data/demo/frame-%06d.depth.png" % (vis_param.index), -1).astype(float)
	depth_im /= 1000.
	depth_im[depth_im == 65.535] = 0

	cam_pose = np.loadtxt("data/demo/frame-%06d.pose.txt" % (vis_param.index))
	
    # Update camera transform with regarded to previous frame
	transform_mat = np.dot(cam_pose, np.linalg.inv(vis_param.current_pose))
	camera.transform(transform_mat)
	vis_param.current_pose = cam_pose

	# TODO: View camera control, view follow the camera pose
	# view_cam = vis_param.view_control.convert_to_pinhole_camera_parameters()
	# view_cam.extrinsic = cam_pose
	# vis_param.view_control.convert_from_pinhole_camera_parameters(view_cam)

	print("Camera Pose: {}".format(camera.get_center()))
	# updateGeometry(camera, "camera_geometry", vis)
	vis.update_geometry(camera)
	vis.update_renderer()

	# Integrate observation into voxel volume (assume color aligned with depth)
	tsdf_vol.integrate(depth_im, cam_intr, cam_pose, weight=1.)

	if(vis_param.index % 20 == 0):


		# verts, faces, norms, colors = tsdf_vol.get_mesh()
		# TSDFVolume.meshwrite("mesh/{}.ply".format(vis_param.index), verts, faces, norms, colors)
		tsdf_vol.extract_mesh()
		updateGeometry(tsdf_vol.get_mesh(), "mesh", vis)

	vis_param.n_steps_left -= 1
	vis_param.index += 1

	return True


def play(vis):
	vis_param.n_steps_left += 10000
	return False

def step(vis):
	vis_param.n_steps_left = 1
	return False


if __name__ == "__main__":


	# ======================================================================================================== #
	# (Optional) This is an example of how to compute the 3D bounds
	# in world coordinates of the convex hull of all camera view
	# frustums in the dataset
	# ======================================================================================================== #
	# print("Estimating voxel volume bounds...")

	cam_intr = np.loadtxt("data/demo/camera-intrinsics.txt", delimiter=' ')
	# vol_bnds = np.zeros((3, 2))
	# for i in range(vis_param.n_imgs):
	# 	# Read depth image and camera pose
	# 	depth_im = cv2.imread("data/frame-%06d.depth.png" % (i), -1).astype(float)
	# 	depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
	# 	depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
	# 	cam_pose = np.loadtxt("data/frame-%06d.pose.txt" % (i))  # 4x4 rigid transformation matrix

	# 	# Compute camera view frustum and extend convex hull
	# 	view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
	# 	vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
	# 	vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))


	# Initialize voxel volume
	print("Initializing voxel volume...")
	vol_bnds = [
		[-4.22106438,  3.86798203],
		[-2.6663104 ,  2.60146141],
		[0.         ,  5.76272371]
	]
	tsdf_vol = TSDFVolume.TSDFVolume(vol_bnds, vox_size=0.02, use_gpu=True, verbose=True)

	window = o3d.visualization.VisualizerWithKeyCallback()
	# window = o3d.visualization.Visualizer()
	window.create_window(window_name="TSDF", width=1280, height=720, visible=True)
	window.get_render_option().mesh_show_back_face = True
	# ctr = window.get_view_control()
	# ctr.translate(0,0,-5.0)

	window.register_key_callback(key=ord(" "), callback_func=step)
	window.register_key_callback(key=ord("."), callback_func=play)

	# TODO: View camera control, view follow the camera pose
	# vis_param.view_control = window.get_view_control()
	# view_cam = vis_param.view_control.convert_to_pinhole_camera_parameters()
	# view_cam.extrinsic = np.eye(4)
	# vis_param.view_control.convert_from_pinhole_camera_parameters(view_cam)

	origin = o3d.geometry.TriangleMesh.create_coordinate_frame()
	camera = visualize.camera(np.eye(4), scale=0.1)
	bbox = visualize.wireframe_bbox([-5., -5., -5.], [5., 5., 5.])

	# grid = visualize.grid()
	# grid.translate((0., -5., 0.))

	window.add_geometry(bbox)
	window.remove_geometry(bbox, reset_bounding_box=False)
	window.add_geometry(origin)
	updateGeometry(camera, "camera_geometry", window)
	# updateGeometry(grid, "grid", window)

	# view_control.set_lookat([0,0,1])
	# vis_param.view_control.set_zoom(0.1)
	# vis_param.view_control.camera_local_translate(1,0,0)
	# view_control.set_up([0,0,1])

	window.register_animation_callback(callback_func=drawNextFrame)
	window.run()

	window.destroy_window()