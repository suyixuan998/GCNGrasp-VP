from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
import sklearn

from . import o3d_extension as o3de


def lift(Teb: np.ndarray,
         d: float) -> np.ndarray:
    Teb = Teb.copy()
    Teb[..., 2, 3] += d
    return Teb


def pregrasp(Teb: np.ndarray,
             d: float) -> np.ndarray:
    Teb = Teb.copy()
    direct = Teb[..., :3, 2]
    direct[..., 2] = np.minimum(direct[..., 2], 0)
    direct /= np.linalg.norm(direct, axis=-1, keepdims=True)
    Teb[..., :3, 3] -= d * direct
    return Teb


@dataclass
class TwoFingerGripper:
    len_finger: float
    len_handle: float
    thickness: float

    def __post_init__(self):
        width = .5
        self._connection = np.array([[2, 3], [0, 1], [2, 4], [3, 5]])
        self._control_pts = np.array(
            [[0, 0, -self.len_handle], [0, 0, 0], [-width, 0, 0], [width, 0, 0],
             [-width, 0, self.len_finger], [width, 0, self.len_finger]], dtype=np.float32
        )
        self._control_pts = np.concatenate([self._control_pts, np.ones_like(self._control_pts[:, :1])], axis=1)

    def o3dmesh(self,
                width: float) -> o3d.geometry.TriangleMesh:
        """ Create an Open3D TriangleMesh gripper. """
        gripper_pcd = self.pcd(width)[:, :3]
        conn = self._connection.copy()

        i = 0
        pts = gripper_pcd[conn[i]]
        pts[:, 0] += self.thickness / 2 * np.array([-1, 1])

        n = len(gripper_pcd)
        conn[i] = [n, n + 1]
        gripper_pcd = np.concatenate([gripper_pcd, pts], axis=0)

        mesh = o3de.create_lineset(gripper_pcd, self.thickness, conn)
        mesh = sum(mesh[1:], mesh[0])
        mesh.compute_vertex_normals()
        return mesh

    def o3dmesh_batch(self,
                      Tgc: np.ndarray,
                      width: np.ndarray):
        return [self.o3dmesh(width[i]).transform(Tgc[i]) for i in range(len(width))]

    def pcd(self,
            width: float) -> np.ndarray:
        ret = self._control_pts.copy()
        ret[:, 0] *= width
        return ret


@dataclass
class TwoFingerGrasp:
    Tgx: np.ndarray
    width: np.ndarray
    score: np.ndarray
    gripper: TwoFingerGripper = None

    def __post_init__(self):
        self.Tgx = self.Tgx.reshape(-1, 4, 4).astype(np.float32)
        self.width = self.width.reshape(-1).astype(np.float32)
        self.score = self.score.reshape(-1).astype(np.float32)

    def __len__(self):
        return len(self.Tgx)

    def __add__(self, other):
        return self.__class__(
            np.concatenate([self.Tgx, other.Tgx], axis=0),
            np.concatenate([self.width, other.width]),
            np.concatenate([self.score, other.score]),
            gripper=self.gripper
        )

    def __getitem__(self, item):
        return self.__class__(self.Tgx[item], self.width[item], self.score[item], gripper=self.gripper)

    # attributes
    @property
    def origin(self) -> np.ndarray:
        return self.Tgx[:, :3, 3]

    @property
    def grip_center(self) -> np.ndarray:
        return self.origin + self.gripper.len_finger * self.Tgx[:, :3, 2]

    @property
    def left_finger_tip(self):
        return self.grip_center - self.width[:, None] / 2 * self.Tgx[:, :3, 0]

    @property
    def right_finger_tip(self):
        return self.grip_center + self.width[:, None] / 2 * self.Tgx[:, :3, 0]

    # methods
    def sphere_group(self,
                     radius: float,
                     theta: float) -> list[np.ndarray]:
        candidates = np.argsort(self.score)[::-1]
        groups = []

        def get_mask(Tgx: np.ndarray):
            pos, approach = Tgx[:, :3, 3], Tgx[:, :3, 2]
            return (np.linalg.norm(pos - pos[0], axis=-1) <= radius) & (approach @ approach[0] > np.cos(theta))

        while len(candidates):
            mask = get_mask(self[candidates].Tgx)
            groups.append(candidates[mask])
            candidates = candidates[~mask]
        return groups

    def mask_by_contact(self,
                        pcd: np.ndarray,
                        radius: float,
                        mode: str = "lr") -> np.ndarray:
        """ Filter grasps based on the distance to the contact points in the point cloud """
        assert mode in ["l", "r", "lr"]
        nn = sklearn.neighbors.NearestNeighbors()
        nn.fit(pcd)

        mask = np.ones(len(self), dtype=bool)
        if "l" in mode:
            mask &= nn.kneighbors(self.left_finger_tip, n_neighbors=1)[0].flatten() <= radius
        if "r" in mode:
            mask &= nn.kneighbors(self.right_finger_tip, n_neighbors=1)[0].flatten() <= radius
        return mask

    def mask_by_percentile(self,
                           percentile: float) -> np.ndarray:
        """ Filter grasps based on the score """
        return self.score >= np.percentile(self.score, percentile)

    def mask_by_relative_thresh(self,
                                thresh: float) -> np.ndarray:
        return self.score >= self.score.max() * thresh

    def nearest_search(self,
                       other: "TwoFingerGrasp") -> np.ndarray:
        dist_gc = np.linalg.norm(self.grip_center[:, None] - other.grip_center, axis=-1)
        dist_org = np.linalg.norm(self.origin[:, None] - other.origin, axis=-1)
        return np.argmin(dist_gc + dist_org, axis=0)

    def nms(self,
            radius: float,
            theta: float = .7854) -> "TwoFingerGrasp":
        """ Non-Maximum Suppression (NMS) based on the grip center """
        return self[[ids[0] for ids in self.sphere_group(radius, theta)]]

    def to_meshes(self,
                  geo_arm: o3d.geometry.Geometry3D = None,
                  cmap: str = "coolwarm") -> list[o3d.geometry.TriangleMesh]:
        """ Get meshes of the grasp poses."""
        if not self: return
        score = self.score - self.score.min() * (len(self) > 1)
        score = score / score.max()

        # gripper meshes
        meshes = [self.gripper.o3dmesh(w).transform(Tgx) for Tgx, w in zip(self.Tgx, self.width)]
        meshes = o3de.paint_by_cmap(meshes, score, cmap)

        # robot arm meshes
        if geo_arm:
            for Tgx in self.Tgx:
                meshes.append(geo_arm.__copy__().transform(Tgx))
        return meshes

    def to_pcd(self):
        pcd = np.stack([self.gripper.pcd(w) for w in self.width])
        return pcd @ self.Tgx.transpose(0, 2, 1)[..., :-1]

    def transform(self,
                  T: np.ndarray) -> "TwoFingerGrasp":
        return self.__class__(T @ self.Tgx, self.width, self.score, gripper=self.gripper)
