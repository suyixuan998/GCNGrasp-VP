from __future__ import annotations

import copy
from functools import cached_property

import cv2
import numpy as np
import open3d as o3d
import pytorch_lightning as pl
from tqdm import tqdm

from .data_type.action import Action
from .data_type.workspace import IrregularWorkspace
from ..api import Path
from ..api.utils import grasp_pose
from ..api.utils import o3d_extension as o3de
from ..api.utils import pose_tf
from ..api.utils.camera import Frame, Pinhole, ObservationSequence


def frame_from_folder(folder: Path) -> tuple[Frame, list[str]]:
    cam, *args = folder.name.split("--")
    frame = Frame(
        camera=Pinhole.from_yaml(f"config/system/{cam}.yaml"),
        color=next(folder.glob("color.*")),
        depth=next(folder.glob("depth.*"))
    )
    return frame, args


class VirtualRobot(ObservationSequence):

    @cached_property
    def config(self) -> dict:
        return Path(self.root / "config.yaml").yaml()

    @cached_property
    def frames(self) -> list[Frame]:
        assert len(self), "Empty dataset"
        frame_folder = self.root / "frames"

        try:
            frames = [Path(f).binary() for f in frame_folder.iterdir()]
            assert len(frames), "No frames found"
            assert len(frames) == len(self), "Fail to load frames"
            return frames

        except Exception as e:
            print(f"Start extracting frames: {e}")
            from .scene import extract_frame_metadata
            from ..api.depth_anything import DepthAnythingV3
            from ..api.utils import Timer

            Timer.set_momentum(.9)
            Timer.reset()
            pl.seed_everything(0)

            # DA3 reconstruction
            da3 = DepthAnythingV3(size="g")
            sa = da3.mapping(self.load(), align_mode="first")
            sa.pcd_processor = self.pcd_processor
            frames = sa.frames_rel

            # visualize
            pcd_abs = sa.pcd_abs
            pcd_abs.paint_uniform_color([0., 0., 1.])
            o3d.visualization.draw_geometries([sa.pcd_rel])

            # cache metadata
            frame_folder.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(tqdm(frames, desc="Extracting frames")):
                extract_frame_metadata(f, target_obj=self.config["target_obj"], pcd_processor=self.pcd_processor)
                Path(frame_folder / f"{f.t}.bin").binary(f)

            Timer.to_dict(log=True)
            Timer.reset()
            return frames

    @cached_property
    def grasps(self) -> grasp_pose.TwoFingerGrasp:

        def fgetter():
            grasps = [f.metadata["grasps"] for f in self.frames]
            grasps: grasp_pose.TwoFingerGrasp = sum(grasps[1:], grasps[0])
            grasps = grasps[grasps.mask_by_contact(np.asarray(self.pcd_obj.points), radius=.01, mode="l")]
            grasps = grasps.nms(.03)
            return grasps

        file = Path(self.root / "grasps.bin")
        return file.lazy_obj(fgetter)

    @property
    def kwargs_scene(self) -> dict:
        return dict(
            target_obj=self.config["target_obj"],
            tog_model_args=self.config["tog_model_args"],
            workspace=self.workspace,
        )

    def load(self) -> list[Frame]:
        camera = Pinhole.from_yaml(self.root / "camera.yaml")
        ret = []
        for data in super().load():
            pose_tf.TFmanager.register(**data["transform"])
            frame = Frame(
                camera=camera,
                color=data["color"],
                depth=data["depth"],
                Tcw=pose_tf.TFmanager["camera", "world"],
                t=data["t"]
            )
            assert frame.Tcw is not None
            ret.append(frame)
        return ret

    @classmethod
    def load_all_instances(cls,
                           folder: Path,
                           check_attr: bool = True) -> list["VirtualRobot"]:
        ret = []
        for p in folder.rglob("camera.yaml"):
            robot = VirtualRobot(root=p.parent)
            ret.append(robot)

            if check_attr: _cache = robot.frames, robot.tog_label
        return ret

    @cached_property
    def obs_obj(self) -> list[o3d.geometry.PointCloud]:
        obj_pcd = [f.to_pcd(mask=f.metadata["dets"].mask[0]) for f in self.frames]
        return list(map(self.pcd_processor, obj_pcd))

    @cached_property
    def pcd_obj(self) -> o3d.geometry.PointCloud:
        file = self.root / "pcd_obj.ply"
        if file.exists():
            return o3d.io.read_point_cloud(file)
        else:
            ret = self.obs_obj
            ret: o3d.geometry.PointCloud = sum(ret[1:], ret[0])
            ret = ret.voxel_down_sample(.005)
            o3d.io.write_point_cloud(file, ret)
            return ret

    @cached_property
    def pcd_processor(self) -> o3de.PointCloudProcessor:
        return o3de.PointCloudProcessor(**self.config["pcd_processor"])

    def set_cur_frame(self, i, seed=None):
        act = Action(Tcw=self.frames[i].Tcw, grasp=False)
        self.move(act.Teb)

        if seed is not None: pl.seed_everything(seed, verbose=False)
        return self.frames[i]

    @cached_property
    def tog_label(self) -> grasp_pose.TwoFingerGrasp:

        def fgetter():
            folder = file.parent
            folder.mkdir(parents=True, exist_ok=True)
            grasps = copy.copy(self.grasps)
            print(f"num grasps: {len(grasps)}")

            # visualize
            img_dir = folder / "visualize"
            img_dir.mkdir(parents=True)

            meshes = grasps.to_meshes()
            img = o3de.auto_render_color(img_size=[512] * 2, geometries=meshes + [self.pcd_obj])
            cv2.imwrite(img_dir / "all.png", img[..., ::-1])
            print(f"save visualize images to `{img_dir}`...")

            for i, m in enumerate(meshes):
                img = o3de.auto_render_color(img_size=[512] * 2, geometries=[m, self.pcd_obj])
                print(i), o3d.visualization.draw_geometries([m, self.pcd_obj])
                cv2.imwrite(img_dir / f"{i}.png", img[..., ::-1])

            # get annotations
            grasps.score = np.zeros(len(grasps))
            while True:
                try:
                    pos_labels = list(map(int, input("pos_labels (e.g., 0 1 2 3): ").split()))
                    grasps.score[pos_labels] = 1.
                    break
                except Exception as e:
                    print(f"Invalid input: {e}")
            return grasps

        file = Path(self.root / "grasps" / "tog_label.bin")
        return file.lazy_obj(fgetter)

    @cached_property
    def workspace(self) -> IrregularWorkspace:
        Tcw = np.stack([f.Tcw for f in self.frames])
        return IrregularWorkspace(pcd=Tcw[:, :3, 3], r_decay=1., is_border=False)

    # Robot interface
    def get_frame(self) -> Frame:
        Tcw_cur = pose_tf.TFmanager["camera", "world"]
        Tcws = np.stack([f.Tcw for f in self.frames])

        diff_lk = np.linalg.norm(Tcws[:, :3, 2] - Tcw_cur[:3, 2], axis=-1)
        diff_pos = np.linalg.norm(Tcws[:, :3, 3] - Tcw_cur[:3, 3], axis=-1)

        i = np.argmin(diff_lk + diff_pos)
        return self.set_cur_frame(i)

    def Teb(self):
        return pose_tf.TFmanager["hand", "base"]

    def move(self, Teb: np.ndarray):
        pose_tf.TFmanager.register(Teb, "hand", "base")
        return True
