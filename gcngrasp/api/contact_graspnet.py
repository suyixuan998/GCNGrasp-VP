from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import utils
from .third_party.contact_graspnet_pytorch import config_utils
from .third_party.contact_graspnet_pytorch.checkpoints import CheckpointIO
from .third_party.contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from .utils import grasp_pose
from .utils import pose_tf
from .utils.camera import Frame

GRIPPER = grasp_pose.TwoFingerGripper(len_finger=.0469, len_handle=.0584, thickness=.003)


class ContactGraspNet:

    def __init__(self):
        """
        Refer to contact_graspnet_pytorch/inference.py for more details.
        """
        config_dir = Path(__file__).parent / "third_party/config"
        self.estimator = GraspEstimator(config_utils.load_config(config_dir / "contact_graspnet", batch_size=1))
        weight_file = utils.download_checkpoint(
            "https://github.com/elchun/contact_graspnet_pytorch/raw/main/checkpoints/contact_graspnet/checkpoints/model.pt",
            filename="contact_graspnet.pt"
        )
        CheckpointIO(checkpoint_dir=weight_file.parent, model=self.estimator.model).load(weight_file.name)

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
