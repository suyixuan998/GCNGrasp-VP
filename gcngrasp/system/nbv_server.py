from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import zmq

from ..api.utils import Path, Timer
from ..api.utils.camera import Frame, FrameWithConfidence
from ..api.utils.zmq_socket import ZmqSocket

try:
    from . import planner
    from .data_type.action import Action
    from .scene import Scene
    from ..api.utils import pose_tf
except:
    pass

ADDR_NBV = "/tmp/gcngraspv2_nbv"


def seed_everything(seed) -> int:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


@dataclass
class NBVProxy:
    config: dict

    def __post_init__(self):
        self.planner = planner.BasePlanner(**self.config)
        self.sock_nbv = ZmqSocket(zmq.REQ, ADDR_NBV, 1)

    def solve(self, scene: "Scene"):
        Tcws = self.planner.sample_lookat(scene, as_transform=True)
        # Tcws = Tcws[~pose_tf.SE3.isin(Tcws, np.stack([f.Tcw for f in scene.frames]), tol=.01)]
        print(f"n candidates: {len(Tcws)}")
        assert len(Tcws) >= 3

        frame = scene.frames[-1]
        frame.metadata.update({
            "fid": len(scene.frames) - 1,
            "cand_Tcws": Tcws,
        })

        Tcw = self.sock_nbv.send(frame, response=True)
        return Action(Tcw, grasp=False)


@dataclass
class NBVServer:
    func_plan_nbv: Callable[[Frame, np.ndarray, int], np.ndarray]  # [frame, cand_Tcws, fid] -> Tcw
    func_reset: Callable[[Frame], None]

    img_resize: list[int] = None
    depth_limit: float = None
    conf_thresh: float = None
    obj_depth_only: bool = False

    def __post_init__(self):
        self.sock_nbv = ZmqSocket(zmq.REP, ADDR_NBV, 1)

    def main(self):
        print("Ready.")
        self.sock_nbv.recv_auto(10, self.process, response=True, thread=False)

    def process(self, frame: Frame) -> np.ndarray:
        # load frame
        if isinstance(frame, str):
            frame = Path(frame).binary()
            frame.metadata.update({"seed": 0, "fid": 0, "cand_Tcws": frame.Tcw[None].repeat(2, axis=0)})

        # process depth
        if self.img_resize:
            frame = frame.resize(self.img_resize)
        if self.depth_limit:
            frame.depth[frame.depth > self.depth_limit] = 0
        if self.conf_thresh:
            frame: FrameWithConfidence
            frame.conf_thresh = 1.05
            frame.depth[~frame.depth_mask] = 0
        if self.obj_depth_only:
            mask = frame.metadata["dets"].mask[0]
            mask = cv2.resize(mask.astype(np.uint8), frame.camera.img_size).astype(bool)
            frame.depth[~mask] = 0

        # plan
        if self.func_plan_nbv:
            cand_Tcws = frame.metadata["cand_Tcws"]

            if frame.metadata["fid"] == 0:
                seed_everything(frame.metadata["seed"])
                self.func_reset(frame)

            with Timer("NBVServer/plan_nbv"):
                return self.func_plan_nbv(frame, cand_Tcws)

        else:
            time.sleep(5)
            Tcw = frame.Tcw.copy()
            Tcw[:3, 3] += np.array([0.5, 0.5, 0.5])
            print("trigger debug mode.")
            return Tcw
