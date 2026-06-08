from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from .third_party.contact_graspnet_pytorch import config_utils
from .third_party.contact_graspnet_pytorch.checkpoints import CheckpointIO
from .third_party.contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from .utils import grasp_pose
from .utils import pose_tf
from .utils.camera import Frame

GRIPPER = grasp_pose.TwoFingerGripper(len_finger=.0469, len_handle=.0584, thickness=.003)


class ContactGraspNet:

    def __init__(self,
                 ckpt_dir: Path = Path("checkpoints/contact_graspnet")):
        """
        Refer to contact_graspnet_pytorch/inference.py for more details.
        :param ckpt_dir: checkpoint directory
        """
        self.estimator = GraspEstimator(config_utils.load_config(ckpt_dir, batch_size=1))
        CheckpointIO(checkpoint_dir=str(ckpt_dir / "checkpoints"), model=self.estimator.model).load('model.pt')

    def __call__(self,
                 pcd: np.ndarray,
                 Tcw: np.ndarray = None) -> grasp_pose.TwoFingerGrasp:
        """
        :param pcd: point cloud with shape (N, 3)
        """
        if Tcw is not None:
            pcd = pose_tf.SE3.transform_pcd(np.linalg.inv(Tcw), pcd)

        Tgc, score, _lft, w = (x[-1] for x in self.estimator.predict_scene_grasps(pcd))
        Tgc[:, :3, 3] += Tgc[:, :3, 2] * GRIPPER.len_finger
        if Tcw is not None: Tgc = Tcw @ Tgc

        gposes = grasp_pose.TwoFingerGrasp(Tgc, w, score, gripper=GRIPPER)
        return gposes[np.argsort(gposes.score)[::-1]]

    def process_frame(self,
                      frame: Frame,
                      max_depth: float = 1.2,
                      xyzr: np.ndarray = None,
                      pcd_processor: Callable = None) -> grasp_pose.TwoFingerGrasp:
        pcd = frame.to_pcd(mask=frame.depth <= max_depth, to_world=False)
        if pcd_processor: pcd = pcd_processor(pcd)
        pcd = np.asarray(pcd.points).astype(np.float32)

        if xyzr is not None:
            r = np.linalg.norm(pcd - xyzr[:3], axis=-1)
            pcd = pcd[r <= xyzr[3]]

        return self(pcd).transform(frame.Tcw)
