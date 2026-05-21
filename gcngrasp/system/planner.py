from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_type.action import Action
from .scene import Scene
from ..api.utils import Timer
from ..api.utils import o3d_extension as o3de
from ..api.utils import pose_tf


@dataclass
class BasePlanner:
    border_margin: float
    max_stride: float
    min_stride: float
    n_candidates: int
    dist_obs: float
    elev_range: tuple[float, float]

    @staticmethod
    def from_config(config: dict) -> "BasePlanner":
        config = config.copy()
        PlannerType = eval(config.pop("type"))
        return PlannerType(**config)

    def sample_lookat(self,
                      scene: Scene,
                      min_stride: float = None,
                      max_stride: float = None,
                      as_transform: bool = False) -> np.ndarray:
        """ Sample a lookat point. """
        field_obj = scene.tog_poses[1]
        dist_obs = max(self.dist_obs, scene.camera.dist_observe(field_obj.radius, border_margin=self.border_margin))

        # filter by stride
        xs = scene.workspace.sample_lookat(
            focus=field_obj.center, dist_min=dist_obs, elev_range=self.elev_range)
        strides = np.linalg.norm(xs - scene.Tcw[:3, 3], axis=-1)
        min_stride = min_stride if min_stride is not None else self.min_stride
        max_stride = max_stride if max_stride is not None else self.max_stride
        xs = xs[(strides <= max_stride) & (strides >= min_stride)]

        # sample
        if self.n_candidates < len(xs):
            xs = xs[o3de.farthest_point_sampling(xs, self.n_candidates)]

        # convert
        if as_transform:
            xs = np.stack([pose_tf.lookat_to_Tcw(x, field_obj.center) for x in xs])
        return xs

    def solve(self,
              scene: Scene,
              visualize: bool = False) -> Action:
        raise NotImplementedError


@dataclass
class ObjectIGViewPlanner(BasePlanner):
    init_stride: float
    init_frames: int
    thresh_ig: float
    loss_weight: dict[str, float]

    def solve(self,
              scene: Scene,
              visualize: bool = False) -> Action:
        # choose a particular field
        field_obj = scene.tog_poses[1]
        field_cluster = scene.tog_poses[1].clusters(thresh_ig=self.thresh_ig, visualize=visualize)
        field_part = field_cluster[0]

        pos = scene.Tcw[:3, 3]
        force_start = len(scene.frames) <= self.init_frames
        with Timer("planner/solve"):

            # scene
            pcd_scene = np.concatenate([field_obj.pcd[np.where(
                np.linalg.norm(field_obj.pcd[:, None] - field_part.pcd, axis=-1).min(axis=-1) > .02
            )[0]], np.asarray(scene.pcd_scene_wo_ground.points)], axis=0)

            # sample candidates
            xs = self.sample_lookat(scene, min_stride=self.init_stride if force_start else None)
            assert len(xs) > 2, "Not enough candidates"
            if not 1.: xs = np.concatenate([xs, pos[None]], axis=0)

            # evaluate
            losses = np.zeros(len(xs), dtype=np.float32)
            for i, x in enumerate(xs):
                loss = {}
                x_lookat = field_part.center - x
                x_lookat_e = x_lookat / np.linalg.norm(x_lookat)

                for k in self.loss_weight:
                    if k == "orient":
                        neg_cossim = field_part.normal @ x_lookat_e
                        loss[k] = 1 + (neg_cossim * field_part.ig).sum() / field_part.ig.sum()

                    elif k == "elev":
                        loss[k] = np.abs(x_lookat[2]) / np.linalg.norm(x_lookat[:2])

                    elif k == "stride":
                        loss[k] = np.linalg.norm(pos - x)

                    elif k == "occ":
                        mask = o3de.cylinder_query(
                            pcd_scene, x,
                            field_part.center + 2 * x_lookat_e * field_part.radius,
                            r=2 * field_part.radius
                        )
                        p_s = pcd_scene[mask]

                        if len(p_s):
                            lk_x2o = field_part.pcd - x
                            lk_x2o /= np.linalg.norm(lk_x2o, axis=-1, keepdims=True)

                            lk_s2o = field_part.pcd - p_s[:, None]
                            d_s2o = np.linalg.norm(lk_s2o, axis=-1)
                            lk_s2o /= d_s2o[..., None]

                            cos_sox = (lk_s2o * lk_x2o).sum(axis=-1)
                            proj = (d_s2o * np.sqrt(1 - np.square(cos_sox))).min(axis=0)
                            inv_proj = 1 / (1 + proj * 1e3)
                            loss[k] = (inv_proj * field_part.ig).sum() / field_part.ig.sum()

                    else:
                        raise ValueError(f"Unknown loss: {k}")

                # print(loss)
                losses[i] = sum(v * self.loss_weight[k] for k, v in loss.items())

            # select
            i = np.argmin(losses)
            if not force_start and i == len(xs) - 1: return
            return Action.from_lookat(xs[i], focus=field_obj.center)
