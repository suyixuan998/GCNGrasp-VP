from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import sklearn.neighbors

from .api import FUNC_API
from .data_type.field import BaseIGField
from .data_type.workspace import IrregularWorkspace
from ..api.utils import Path, Timer, clean_cached_properties
from ..api.utils import grasp_pose
from ..api.utils import o3d_extension as o3de
from ..api.utils import sv_extension as sve
from ..api.utils.camera import Frame, Pinhole


def extract_frame_metadata(frame: Frame,
                           target_obj: str = None,
                           pcd_processor: Callable = None):
    # extract target object
    dets = frame.metadata.get("dets")
    if target_obj and not dets:
        with Timer("api/Grounded-SAM2"):
            dets = FUNC_API("gsam", frame.cvt_color("BGR"), target_obj)
        frame.metadata["dets"] = dets = sve.refine_masks_by_ns(dets, frame.pointmap, ksize=3, radius=0.02, nb_neighbors=7)
        assert len(dets) > 0

    if pcd_processor:
        # region of interest
        pcd = frame.to_pcd(mask=dets.mask[0])
        if pcd_processor: pcd = pcd_processor(pcd)
        pcd = get_pcd_array(pcd)
        center = pcd.mean(axis=0)
        roi = np.append(
            (frame.Twc @ np.append(center, 1))[:3],
            2 * np.linalg.norm(pcd - center, axis=-1).max()
        )

        # extract grasps
        if not frame.metadata.get("grasps"):
            with Timer("api/ContactGraspNet"):
                frame.metadata["grasps"] = FUNC_API("cgn", frame, xyzr=roi, pcd_processor=pcd_processor)


def get_pcd_array(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    return np.asarray(pcd.points).astype(np.float32)


@dataclass
class Scene:
    config: dict = field(repr=False)
    target_obj: str
    tog_model_args: tuple = field(default_factory=tuple)
    workspace: IrregularWorkspace = None

    pcd_obj: o3d.geometry.PointCloud = field(init=False, default_factory=o3d.geometry.PointCloud)
    pcd_scene: o3d.geometry.PointCloud = field(init=False, default_factory=o3d.geometry.PointCloud)
    frames: list[Frame] = field(init=False, default_factory=list)

    def __post_init__(self):
        config = self.config["scene"]
        self.noisy_pcd_processor = o3de.PointCloudProcessor(
            nn_radius=config["nn_radius"], nn_k_thresh=config["nn_k_thresh"])
        self.full_pcd_processor = o3de.PointCloudProcessor(voxel_size=config["voxel_size"])

    def add_pcd(self,
                frame: Frame):
        mask = frame.metadata["dets"].mask[0]
        pcd_obj, pcd_scene = map(self.noisy_pcd_processor, (frame.to_pcd(mask), frame.to_pcd(~mask)))
        self.pcd_obj += pcd_obj
        self.pcd_scene += pcd_scene

    @property
    def camera(self) -> Pinhole:
        return self.frames[-1].camera

    @cached_property
    def grasp_poses(self) -> grasp_pose.TwoFingerGrasp:
        # gps = [f.metadata["grasps"] for f in self.frames]
        # gps: grasp_pose.TwoFingerGrasp = sum(gps[1:], gps[0]) if len(gps) > 1 else gps[0]
        gps = self.frames[-1].metadata["grasps"]
        gps = gps[gps.mask_by_contact(get_pcd_array(self.pcd_obj), radius=.02, mode="l")]
        gps.width += self.config["scene"]["grasp_w_offset"]
        return gps

    def integrate(self,
                  frame: Frame,
                  mapper: Callable = None,
                  check_map: bool = False) -> Frame:
        """ Integrate a new frame into the scene """
        if np.all(frame.Tcw == self.Tcw): raise ValueError("Same frame")
        self.frames.append(frame)
        extract_frame_metadata(frame, target_obj=self.target_obj)

        # mapping
        if mapper:
            self.pcd_obj.clear()
            self.pcd_scene.clear()
            mask = frame.metadata["dets"].mask[0]
            frame.depth[~mask] = 0.

            with Timer("scene/mapping"):
                *frames_prev, frame = mapper(self.frames)
                for f in frames_prev: self.add_pcd(f)

            # check
            if check_map:
                pcd_abs = self.frames[-1].to_pcd()
                pcd_abs.paint_uniform_color([0, 0, 1.])
                o3d.visualization.draw_geometries([pcd_abs, frame.to_pcd()])

        extract_frame_metadata(frame, pcd_processor=self.noisy_pcd_processor)

        with Timer("scene/process_pcd"):
            self.add_pcd(frame)
            self.pcd_obj, self.pcd_scene = map(self.full_pcd_processor, (self.pcd_obj, self.pcd_scene))

            # filter out points inside the object
            pcd_obj, pcd_scene = map(get_pcd_array, (self.pcd_obj, self.pcd_scene))
            nn_obj = sklearn.neighbors.NearestNeighbors(n_neighbors=1)
            nn_obj.fit(pcd_obj)
            dist = nn_obj.kneighbors(pcd_scene)[0].flatten()
            self.pcd_scene = self.pcd_scene.select_by_index(np.where(dist > .02)[0])

            # clean cache
            clean_cached_properties(self, ["grasp_poses", "tog_poses"])
            return frame

    def main(self,
             frame_getter: Callable):
        """ Main function """
        while True:
            frame = frame_getter()
            if not frame: break
            yield self.integrate(frame)

    @property
    def pcd_scene_wo_ground(self) -> o3d.geometry.PointCloud:
        i = np.where(get_pcd_array(self.pcd_scene)[:, 2] > self.z_ground)[0]
        return self.pcd_scene.select_by_index(i)

    @property
    def Tcw(self) -> np.ndarray:
        return self.frames[-1].Tcw if self.frames else None

    def save_info(self,
                  folder: Path,
                  **opt_o3d):
        folder.mkdir(parents=True, exist_ok=True)
        self.to_o3d(**opt_o3d, save_file=folder / "visualize.glb")
        cv2.imwrite(str(folder / f"color.png"), self.frames[-1].cvt_color("BGR"))
        Path(folder / "timer.yaml").yaml(Timer.to_dict(log=False))

    @cached_property
    def tog_poses(self) -> tuple[grasp_pose.TwoFingerGrasp, BaseIGField]:
        grasp = self.grasp_poses.nms(radius=self.config["scene"]["grasp_nms_r"])
        assert len(grasp)

        with Timer("api/TaskOrientedGraspNet"):
            score_tog, field = FUNC_API(
                "tog", get_pcd_array(self.pcd_obj), grasp.to_pcd(), *self.tog_model_args)

        w_tog = self.config["weight_tog"]
        # score_tog /= score_tog.max()
        grasp.score = score_tog * w_tog + grasp.score * (1 - w_tog)
        # grasp.score *= grasp.score
        grasp = grasp[np.argsort(grasp.score)[::-1]]
        return grasp, field

    def to_o3d(self,
               obj: bool = True,
               scene: bool = True,
               tog: bool = True,
               field: bool = True,
               traj: bool = True,
               coord: bool = False,
               next_Tcw: np.ndarray = None,
               save_file: str = None) -> list[o3d.geometry.Geometry3D]:
        geos = []

        if tog:
            g, f = self.tog_poses
            g = g.nms(radius=self.config["scene"]["grasp_nms_r"])
            geos.extend(g.to_meshes(cmap="coolwarm"))
            if f and field:
                geos.append(f.to_o3d())
                obj = False

        if scene: geos.append(self.pcd_scene)
        if obj: geos.append(self.pcd_obj)
        if coord: geos.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))

        if traj:
            Tcw = np.stack([f.Tcw for f in self.frames])
            p = np.linspace(0, 1, len(Tcw))

            cmap = plt.get_cmap("hot_r")
            color_near = cmap(p / 2)[:, :3]
            color_far = cmap(p / 2 + 0.5)[:, :3]

            for i, T in enumerate(Tcw):
                geos.append(o3de.create_camera(
                    T, width=self.config["scene"]["vis_cam_width"],
                    color_near=color_near[i], color_far=color_far[i]
                ))

            if Tcw.shape[0] > 1:
                geos.extend(o3de.create_traj(
                    Tcw[:, :3, 3], thickness=self.config["scene"]["vis_traj_thickness"], cmap="hot_r"))

        if next_Tcw is not None:
            geos.append(o3de.create_camera(next_Tcw, width=self.config["scene"]["vis_cam_width"]))

        if save_file:
            tri = o3de.concat_to_trimesh(geos)
            tri.export(save_file)

        return geos

    @property
    def z_ground(self) -> float:
        return get_pcd_array(self.pcd_obj)[:, 2].min().item()
