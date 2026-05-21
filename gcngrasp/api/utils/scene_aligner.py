from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import open3d as o3d

from . import pose_tf
from .camera import Frame


def align_depth(pred: np.ndarray,
                depth: np.ndarray,
                mask: np.ndarray,
                bias: bool = False):
    assert depth.ndim == 2, f"Depth map should be 2D, but got {depth.ndim}D"
    d1, d2 = 1 / pred[mask], 1 / depth[mask]

    if bias:
        d1m, d2m = d1.mean(), d2.mean()
        d1_, d2_ = d1 - d1m, d2 - d2m
        s = (d1_ @ d2_) / (d1_ @ d1_)
        b = d2m - s * d1m
        pred = 1 / (s / pred + b)

    else:
        s = 1 / ((d1 @ d2) / (d1 @ d1))
        pred = pred * s

    return pred


@dataclass
class SceneAligner:
    frames_abs: list[Frame] = field(default_factory=list)
    frames_rel: list[Frame] = field(default_factory=list)
    pcd_processor: Callable = None
    metadata: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.frames_abs)

    def align_frames(self, mode: str):
        ids = self.get_align_ids(mode)

        if len(ids) == 1:
            i = ids[0]
            f1, f2 = self.frames_abs[i], self.frames_rel[i]
            assert f1.camera.img_size == f2.camera.img_size, "Please run `resize_rel_frames` first"

            # scale depth
            mask = f1.depth_mask & f2.depth_mask
            d1, d2 = 1 / f1.depth[mask], 1 / f2.depth[mask]
            s = 1 / ((d1 @ d2) / (d2 @ d2))
            for f in self.frames_rel:
                f.depth *= s
                f.Tcw[:3, 3] *= s

            # corrected transform
            Tmw = self.frames_abs[i].Tcw @ self.frames_rel[i].Twc
            for f in self.frames_rel: f.Tcw = Tmw @ f.Tcw

        else:
            p1 = np.stack([f.Tcw[:3, 3] for f in self.frames_abs])
            assert p1.std(axis=0).max() > 1e-3, "bad poses"
            p2 = np.stack([f.Tcw[:3, 3] for f in self.frames_rel])

            # corrected transform
            Tmw = pose_tf.align_pcd(p2, p1, method="se3")
            s = np.linalg.det(Tmw[:3, :3]) ** (1 / 3)

            for f in self.frames_rel:
                f.depth *= s
                f.Tcw = Tmw @ f.Tcw
                f.Tcw[:3, :3] = pose_tf.normalize_rotation(f.Tcw[:3, :3])

    def copy_metadata(self):
        for i in range(len(self)):
            fa, fr = self.frames_abs[i], self.frames_rel[i]
            fr.t = fa.t
            fr.metadata = fa.metadata

    def get_align_ids(self, mode: str) -> list[int]:
        return {
            "all": list(range(len(self))),
            "first": [0],
            "last": [-1],
        }[mode]

    def get_aligned_pcd(self, mode: str) -> o3d.geometry.PointCloud:
        """ Align point clouds """
        ids = self.get_align_ids(mode)
        x1, x2 = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()

        for i in ids:
            f1, f2 = self.frames_abs[i], self.frames_rel[i]
            assert f1.camera.img_size == f2.camera.img_size, "Please run `resize_rel_frames` first"

            x1 += f1.to_pcd(f2.depth_mask)
            x2 += f2.to_pcd(f1.depth_mask)

        x1, x2 = map(lambda x: np.asarray(x.points), (x1, x2))
        T = pose_tf.align_pcd(x2, x1, method="linear-reg")
        return self.pcd_rel.transform(T)

    @property
    def pcd_abs(self) -> o3d.geometry.PointCloud:
        pcds = [f.to_pcd() for f in self.frames_abs]
        if self.pcd_processor: pcds = list(map(self.pcd_processor, pcds))
        return sum(pcds[1:], pcds[0])

    @property
    def pcd_rel(self) -> o3d.geometry.PointCloud:
        pcds = [f.to_pcd() for f in self.frames_rel]
        if self.pcd_processor: pcds = list(map(self.pcd_processor, pcds))
        return sum(pcds[1:], pcds[0])

    def resize_rel_frames(self):
        for i, (f1, f2) in enumerate(zip(self.frames_abs, self.frames_rel)):
            self.frames_rel[i] = f2.resize(f1.camera.img_size)
