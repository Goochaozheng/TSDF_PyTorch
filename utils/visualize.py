import open3d as o3d
import numpy as np
import matplotlib.cm

# from utils.geoUtils import Isometry

def camera(transform: list, wh_ratio: float = 4.0 / 3.0, scale: float = 1.0, fovx: float = 90.0):
    """
    A camera frustum for visualization.

    :param transform: The initial tansform of the geometry object.

    """

    if transform.shape != (4,4):
        print("Transform Matrix must be 4x4, but get {}".format(transform.shape))
        raise Exception()
        
    pw = np.tan(np.deg2rad(fovx / 2.)) * scale
    ph = pw / wh_ratio
    all_points = np.asarray([
        [0.0, 0.0, 0.0],
        [pw, ph, scale],
        [pw, -ph, scale],
        [-pw, ph, scale],
        [-pw, -ph, scale],
    ])

    line_indices = np.asarray([
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [1, 3], [3, 4], [2, 4]
    ])

    geom = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(all_points),
        lines=o3d.utility.Vector2iVector(line_indices))

    my_color = np.asarray(matplotlib.cm.get_cmap('tab10').colors)[3, :3]
    geom.colors = o3d.utility.Vector3dVector(np.repeat(np.expand_dims(my_color, 0), line_indices.shape[0], 0))

    geom.transform(transform)

    return geom


def grid(transform: list = np.eye(4), wh_ratio: float = 4.0 / 3.0, scale: float = 1.0, fovx: float = 90.0):
    """
    Grid object for visualization.
    """


    vertices_x = [(np.ceil(i/2)-6) * scale for i in range(1, 23)]
    vertices_z = [((i%2==0)*10-6) * scale for i in range(22)]
    vertices = np.ndarray((22, 3))
    vertices[:, 0] = vertices_x
    vertices[:, 1] = np.zeros(22)
    vertices[:, 2] = vertices_z

    line_a = [i*2 for i in range(11)]
    line_b = [i*2+1 for i in range(11)]
    line_indices = np.ndarray((11, 2))
    line_indices[:, 0] = line_a
    line_indices[:, 1] = line_b

    geom = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(line_indices)
    )

    geom.colors = o3d.utility.Vector3dVector(np.repeat(np.expand_dims(np.zeros((3,)), 0), line_indices.shape[0], 0))
    geom.transform(transform)

    return geom


def wireframe_bbox(extent_min=None, extent_max=None, color_id=-1):
    """
    Bounding box.
    """

    if extent_min is None:
        extent_min = [0.0, 0.0, 0.0]
    if extent_max is None:
        extent_max = [1.0, 1.0, 1.0]

    if color_id == -1:
        my_color = np.zeros((3,))
    else:
        my_color = np.asarray(matplotlib.cm.get_cmap('tab10').colors)[color_id, :3]

    all_points = np.asarray([
        [extent_min[0], extent_min[1], extent_min[2]],
        [extent_min[0], extent_min[1], extent_max[2]],
        [extent_min[0], extent_max[1], extent_min[2]],
        [extent_min[0], extent_max[1], extent_max[2]],
        [extent_max[0], extent_min[1], extent_min[2]],
        [extent_max[0], extent_min[1], extent_max[2]],
        [extent_max[0], extent_max[1], extent_min[2]],
        [extent_max[0], extent_max[1], extent_max[2]],
    ])
    line_indices = np.asarray([
        [0, 1], [2, 3], [4, 5], [6, 7],
        [0, 4], [1, 5], [2, 6], [3, 7],
        [0, 2], [4, 6], [1, 3], [5, 7]
    ])
    geom = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(all_points),
        lines=o3d.utility.Vector2iVector(line_indices))
    geom.colors = o3d.utility.Vector3dVector(np.repeat(np.expand_dims(my_color, 0), line_indices.shape[0], 0))

    return geom
