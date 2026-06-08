from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import sklearn
import trimesh

CYLINDER_ENDPOINT = np.zeros([2, 3])
CYLINDER_ENDPOINT[:, 2] = -.5, .5

"""
   5 __________ 4
    /         /|
   /         / |
3 /_________/6 |
  |         |  |
  |  2      |  | 7
  |         | /
0 |_________|/1 
"""
BBOX_LINES = np.array([[0, 1], [0, 2], [0, 3], [1, 6], [1, 7], [2, 5], [2, 7], [3, 5], [3, 6], [4, 5], [4, 6], [4, 7]])


def auto_render_color(img_size: list[int],
                      geometries: list[o3d.geometry.Geometry]) -> np.ndarray:
    """ Render the given geometries to a color image using Open3D's visualizer. """
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_size[0], height=img_size[1], visible=True)

    for geo in geometries: vis.add_geometry(geo)
    img = vis.capture_screen_float_buffer(do_render=True)

    vis.destroy_window()
    return np.round(np.asarray(img) * 255).astype(np.uint8)


def ball_query(pcd: np.ndarray,
               center: np.ndarray,
               r: float) -> np.ndarray:
    return np.linalg.norm(pcd - center, axis=1) <= r


def concat_to_trimesh(geos: list[o3d.geometry.Geometry3D]) -> trimesh.Scene:
    geos = list(map(to_trimesh, geos))
    return trimesh.Scene(geos)


def correct_obb(box: o3d.geometry.OrientedBoundingBox) -> o3d.geometry.OrientedBoundingBox:
    from .pose_tf import rota_toward_identity

    R_pcd = rota_toward_identity(np.asarray(box.R))
    return o3d.geometry.OrientedBoundingBox(
        center=np.asarray(box.center),
        R=box.R @ R_pcd.T,
        extent=box.extent @ np.abs(R_pcd).T
    )


def create_arrow(start: np.ndarray,
                 end: np.ndarray,
                 radius: float = 0.002,
                 color: tuple = (1, 0, 0)) -> o3d.geometry.TriangleMesh:
    """ Create an Open3D TriangleMesh arrow from start to end """
    from .pose_tf import align_pcd

    length = np.linalg.norm(end - start)

    # z-axis arrow
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=radius, cone_radius=radius * 2,
        cylinder_height=max(1e-6, length - radius * 2), cone_height=radius * 5, resolution=20
    )
    arrow.paint_uniform_color(color)

    # rotate the arrow to align with the vector
    arrow.translate([0, 0, -length / 2])
    arrow.transform(align_pcd(CYLINDER_ENDPOINT * length, np.stack([start, end]), method="sim3"))
    return arrow


def create_bbox(bbox: o3d.geometry.OrientedBoundingBox,
                thickness: float) -> o3d.geometry.TriangleMesh:
    lines = create_lineset(np.asarray(bbox.get_box_points()), thickness, lines=BBOX_LINES)
    return sum(lines[1:], lines[0])


def create_camera(Tcw: np.ndarray,
                  width: float,
                  aspect_ratio: float = 0.75,
                  depth_width_ratio: float = 0.6,
                  near_far_ratio: float = 0.3,
                  color_near: np.ndarray = (0.2,) * 3,
                  color_far: np.ndarray = (0.8,) * 3) -> o3d.geometry.TriangleMesh:
    height = width * aspect_ratio
    depth = width * depth_width_ratio

    # initialize vertices for the camera frustum
    vertices = np.stack(list(map(
        np.ravel, np.meshgrid([-width, width], [-height, height]))
    ), axis=-1) / 2
    vertices = np.concatenate([vertices[[0, 1, 3, 2]], np.full([4, 1], depth)], axis=-1)
    vertices = np.concatenate([vertices, vertices * near_far_ratio], axis=0)
    vertices[4:, 2] = 0

    # connect vertices to form triangles
    faces = np.array([[0, 1, 4], [1, 5, 4], [1, 2, 5], [2, 6, 5],
                      [2, 3, 6], [3, 7, 6], [3, 0, 7], [0, 4, 7], [4, 5, 7], [5, 6, 7]])
    faces = np.concatenate([faces, faces[..., ::-1]], axis=0)

    # initialize the TriangleMesh object
    trimesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(faces)
    )
    trimesh.compute_vertex_normals()

    # set colors for the near and far planes
    colors = np.stack([color_far, color_near], axis=0).repeat(4, axis=0)
    trimesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return trimesh.transform(Tcw)


def create_coord(T: np.ndarray = None,
                 size: float = .1) -> o3d.geometry.TriangleMesh:
    T = np.eye(4) if T is None else T
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size).transform(T)


def create_lineset(points: np.ndarray,
                   thickness: float,
                   lines: np.ndarray = None) -> list[o3d.geometry.TriangleMesh]:
    from .pose_tf import align_pcd

    if lines is None:
        lines = np.stack(list(itertools.combinations(np.arange(len(points)), 2)))

    meshes = []
    for pts in points[lines]:
        length = np.linalg.norm(pts[0] - pts[1])
        if length < 1e-6: continue

        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=thickness / 2, height=length)
        mesh.transform(align_pcd(CYLINDER_ENDPOINT * length, pts, method="sim3"))
        meshes.append(mesh)
    return meshes


def create_traj(waypoints: np.ndarray,
                thickness: float,
                cmap: str = "rainbow") -> list[o3d.geometry.TriangleMesh]:
    length = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    percent = np.cumsum(length) / length.sum()

    i = np.arange(len(waypoints))
    meshes = create_lineset(waypoints, thickness, lines=np.stack([i[:-1], i[1:]], axis=-1))
    meshes = paint_by_cmap(meshes, score=percent, cmap=cmap)
    return meshes


def cylinder_query(pcd: np.ndarray,
                   p1: np.ndarray,
                   p2: np.ndarray,
                   r: float = None) -> np.ndarray:
    from .pose_tf import align_pcd, SE3

    v = p2 - p1
    height = np.linalg.norm(v)

    T = align_pcd(np.stack([p1, p2]), CYLINDER_ENDPOINT * height, method="sim3")
    pcd = SE3.transform_pcd(T, pcd)
    mask = np.abs(pcd[:, 2]) <= height / 2

    if r: mask &= np.linalg.norm(pcd[:, :2], axis=1) <= r
    return mask


def estimate_pcd_interval(pcd: np.ndarray,
                          k: int) -> np.ndarray:
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k + 1).fit(pcd)
    distances, _ = nbrs.kneighbors(pcd)
    return distances[:, 1:].mean(axis=1)


def estimate_pcd_normal(pcd: o3d.geometry.PointCloud,
                        cam_pos: np.ndarray = None,
                        search_param: o3d.geometry.KDTreeSearchParam = None
                        ) -> o3d.geometry.PointCloud:
    search_param = search_param or o3d.geometry.KDTreeSearchParamKNN(knn=30)
    pcd.estimate_normals(search_param)
    if cam_pos is not None:
        pcd.orient_normals_towards_camera_location(cam_pos)
        pcd.estimate_normals(search_param)
    return pcd


def farthest_point_sampling(pcd: np.ndarray,
                            num_samples: int | float) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) algorithm.
    :param pcd: The input point cloud.
    :param num_samples: The number of points to sample.
    :return: The sampled indices.
    """
    n = len(pcd)
    if isinstance(num_samples, float): num_samples = int(num_samples * n)
    if num_samples > n: return np.arange(n)

    distances = np.full(n, np.inf)
    indices = np.zeros(num_samples, dtype=int)
    indices[0] = np.random.randint(0, n)

    for i in range(num_samples - 1):
        distances = np.minimum(distances, np.linalg.norm(pcd - pcd[indices[i]], axis=1))
        indices[i + 1] = np.argmax(distances)

    return indices


def from_mesh_array(ver_and_tri: tuple[np.ndarray, np.ndarray]) -> o3d.geometry.TriangleMesh:
    vertices, triangles = ver_and_tri
    return o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(triangles)
    )


def from_pcd_array(pcd_and_rgb: np.ndarray,
                   color_scale: float = 255) -> o3d.geometry.PointCloud:
    """ :param pcd_and_rgb: [N, 6] """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_and_rgb[:, :3]))
    if pcd_and_rgb.shape[-1] >= 6:
        pcd.colors = o3d.utility.Vector3dVector(pcd_and_rgb[:, 3:6] / color_scale)
    return pcd


def from_trimesh(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(mesh.vertices),
        triangles=o3d.utility.Vector3iVector(mesh.faces)
    )

    vc = getattr(mesh.visual, "vertex_colors")
    if vc is not None:
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vc[..., :3] / 255)
    return o3d_mesh


def minimum_2d_bbox_orientation(pcd: np.ndarray) -> np.ndarray:
    from shapely.geometry import MultiPoint

    rect = MultiPoint(pcd).minimum_rotated_rectangle
    corners = np.array(rect.exterior.coords)[:-1]
    R = np.stack([corners[0], corners[2]]) - corners[1]
    R /= np.linalg.norm(R, axis=-1, keepdims=True)
    return R.T


def pcd_combination(pcds: list[np.ndarray],
                    r: int = None,
                    duplicate: bool = True) -> list[np.ndarray]:
    res = pcds.copy()
    for k in range(2, (r or len(pcds)) + 1):
        for x in itertools.combinations(pcds, k):
            y = np.concatenate(x, axis=0)

            if duplicate:
                v = estimate_pcd_interval(x[0], 1).min()
                pcd = from_pcd_array(y).voxel_down_sample(v)
                y = to_pcd_array(pcd)

            res.append(y)
    return res


def pcd_registration(src: o3d.geometry.PointCloud,
                     dst: o3d.geometry.PointCloud,
                     T: np.ndarray = None,
                     radius: float = .05) -> tuple[float, np.ndarray]:
    """
    Point-to-point ICP registration
    :return: fitness, T_src_dst
    """
    o3d_reg = o3d.pipelines.registration
    # coarse
    if T is None:
        src_fpfh, dst_fpfh = map(lambda pcd: o3d_reg.compute_fpfh_feature(
            pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
        ), (src, dst))
        result: o3d_reg.RegistrationResult = o3d_reg.registration_ransac_based_on_feature_matching(
            src, dst, src_fpfh, dst_fpfh, True, radius,
            estimation_method=o3d_reg.TransformationEstimationPointToPoint(),
            checkers=[o3d_reg.CorrespondenceCheckerBasedOnDistance(radius)],
        )
        T = result.transformation
    # fine tune
    result: o3d_reg.RegistrationResult = o3d_reg.registration_icp(
        src, dst, radius, T,
        estimation_method=o3d_reg.TransformationEstimationPointToPoint(),
    )
    return result.fitness, result.transformation.astype(np.float32)


def paint_by_cmap(geos: list[o3d.geometry.Geometry3D],
                  score: np.ndarray,
                  cmap: str) -> list[o3d.geometry.Geometry3D]:
    assert score.ndim == 1
    if isinstance(geos, o3d.geometry.Geometry3D):
        geos = [geos.__copy__() for _ in range(len(score))]

    colors = plt.get_cmap(cmap)(score)[:, :3]
    for g, c in zip(geos, colors):
        g.paint_uniform_color(c)
    return geos


def regularize_pcd_size(pcd: np.ndarray,
                        num_samples: int,
                        use_fps: bool = True) -> np.ndarray:
    m = pcd.shape[0]
    if m < num_samples:
        return np.concatenate([np.arange(m, dtype=int), np.random.choice(m, num_samples - m, replace=True)])
    else:
        return farthest_point_sampling(pcd, num_samples) if use_fps else np.random.choice(m, num_samples, replace=False)


def rgbd2pcd(pointmap: np.ndarray,
             color: np.ndarray,
             mask: np.ndarray,
             pcd: o3d.geometry.PointCloud = None) -> o3d.geometry.PointCloud:
    """
    :param pointmap: 2D dense field of points [H, W, 3]
    :param color: BGR color [H, W, 3]
    :param mask: mask of the points [H, W]
    """
    pcd = pcd or o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointmap[mask])
    pcd.colors = o3d.utility.Vector3dVector(color[mask][..., ::-1] / 255)
    return pcd


def to_mesh_array(mesh) -> tuple[np.ndarray, np.ndarray]:
    mesh_type = (trimesh.Trimesh, o3d.geometry.TriangleMesh).index(type(mesh))
    if mesh_type == 0:
        args = mesh.vertices, mesh.faces
    elif mesh_type == 1:
        args = mesh.vertices, mesh.triangles
    return tuple(map(np.asarray, args))


def to_pcd_array(pcd: o3d.geometry.PointCloud,
                 color_scale: float = 255) -> np.ndarray:
    """ :return: [N, 6] """
    pcd_np = np.asarray(pcd.points)
    if pcd.has_colors():
        pcd_np = np.concatenate([pcd_np, np.asarray(pcd.colors) * color_scale], axis=-1)
    return pcd_np.astype(np.float32)


def to_trimesh(geo) -> trimesh.Trimesh:
    if isinstance(geo, o3d.geometry.TriangleMesh):
        return trimesh.Trimesh(
            vertices=geo.vertices,
            faces=geo.triangles,
            vertex_colors=geo.vertex_colors
        )
    elif isinstance(geo, o3d.geometry.PointCloud):
        return trimesh.PointCloud(
            vertices=geo.points,
            colors=geo.colors
        )
    else:
        raise TypeError(f"Unsupported type: {type(geo)}")


def transform_bbox(T: np.ndarray,
                   bbox: o3d.geometry.OrientedBoundingBox) -> o3d.geometry.OrientedBoundingBox:
    return bbox.rotate(T[:3, :3], center=np.zeros(3)).translate(T[:3, 3])


def zaxis_aligned_bbox(pcd: o3d.geometry.PointCloud) -> o3d.geometry.OrientedBoundingBox:
    pcd = np.asarray(pcd.points, dtype=np.float32)
    R = zaxis_aligned_bbox_orientation(pcd)
    pcd = pcd @ R
    bounds = np.stack([pcd.min(axis=0), pcd.max(axis=0)])
    return o3d.geometry.OrientedBoundingBox(R @ ((bounds[0] + bounds[1]) / 2), R, bounds[1] - bounds[0])


def zaxis_aligned_bbox_orientation(pcd: np.ndarray) -> np.ndarray:
    R = np.eye(3)
    R[:2, :2] = minimum_2d_bbox_orientation(pcd[:, :2])
    return R


@dataclass
class MeshRepository:
    save_dir: Path
    render_size: tuple[int, int] = None

    def __post_init__(self):
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def __iter__(self):
        return (f.stem for f in self.save_dir.glob("*.obj"))

    def get(self, name, fget) -> o3d.geometry.TriangleMesh:
        """ Load or save a mesh by name. """
        f = self.save_dir / f"{name}.obj"
        if f.exists():
            mesh = o3d.io.read_triangle_mesh(f)
        else:
            mesh = fget()
            o3d.io.write_triangle_mesh(f, mesh)

            if self.render_size:
                img = auto_render_color(self.render_size, [mesh])
                o3d.io.write_image(f.with_suffix(".png"), o3d.geometry.Image(img))
        return mesh


@dataclass
class PointCloudProcessor:
    voxel_size: float = None
    nn_radius: float = None
    nn_k_thresh: int = None
    std_ratio: float = None

    def __post_init__(self):
        self.rmr_out_kwargs = dict(radius=self.nn_radius, nb_points=self.nn_k_thresh)

    def __call__(self,
                 pcd: o3d.geometry.PointCloud,
                 R: np.ndarray = None) -> o3d.geometry.PointCloud:
        if self.voxel_size is not None:
            pcd = pcd.voxel_down_sample(self.voxel_size)

        if any(self.rmr_out_kwargs.values()):
            pcd = pcd.remove_radius_outlier(**self.rmr_out_kwargs)[0]

        if self.std_ratio:
            R = pcd.get_minimal_oriented_bounding_box().R if R is None else R
            pcd_np = np.asarray(pcd.points) @ R
            pcd = pcd.select_by_index(np.where(
                np.all(np.abs(pcd_np - pcd_np.mean(axis=0)) < self.std_ratio * pcd_np.std(axis=0), axis=1)
            )[0])
        return pcd


class PinholeRenderer:

    def __init__(self,
                 img_size: list[int],
                 intrinsics: list[float]):
        assert len(img_size) == 2 and len(intrinsics) == 4
        self.cam = o3d.camera.PinholeCameraIntrinsic(*img_size, *intrinsics)

    def __call__(self,
                 geometries: list[o3d.geometry.Geometry],
                 Tcw: np.ndarray = np.eye(4),
                 return_color: bool = True,
                 return_depth: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Render the given geometries from the camera's perspective.
        :param geometries: List of Open3D geometries to render
        :param Tcw: Camera-to-world transformation matrix
        :param return_color: whether to return the color image
        :param return_depth: whether to return the depth image
        :return: (color image, depth image)
        """
        assert return_color or return_depth
        from open3d.visualization import rendering

        rend = rendering.OffscreenRenderer(width=self.cam.width, height=self.cam.height)
        rend.setup_camera(self.cam, np.linalg.inv(Tcw))

        scene: rendering.Open3DScene = rend.scene
        for i, geo in enumerate(geometries):
            scene.add_geometry(f"geo{i}", geo, material=rendering.MaterialRecord())

        ret = [None] * 2
        if return_color: ret[0] = np.round(np.asarray(rend.render_to_image()) * 255).astype(np.uint8)
        if return_depth: ret[1] = np.asarray(rend.render_to_depth_image(z_in_view_space=True))
        return tuple(ret)


if __name__ == '__main__':
    pcds = [np.random.rand(1000, 3) for _ in range(5)]
    pcd_combination(pcds, 2, duplicate=True)
