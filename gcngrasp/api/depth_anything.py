from __future__ import annotations

import warnings
from dataclasses import dataclass

import cv2
import numpy as np
from depth_anything_3.api import DepthAnything3, Prediction

from . import utils
from .utils.camera import Frame, FrameWithConfidence, Pinhole
from .utils.scene_aligner import align_depth, SceneAligner

DEVICE, DTYPE = utils.get_tensor_args()
FRAME_KWARGS = dict(
    conf_thresh=1.01, sensor_depth=False,
)


def resize_mask(mask: np.ndarray,
                img_size: tuple[int, int]):
    return cv2.resize(mask.astype(np.uint8), img_size).astype(bool)


def prediction2frames(pred: Prediction) -> list[FrameWithConfidence]:
    frames = []
    for i, (color, depth, conf, extrinsic, intrinsic) in enumerate(zip(
            pred.processed_images, pred.depth, pred.conf, pred.extrinsics, pred.intrinsics
    )):
        Twc = np.eye(4)
        Twc[:len(extrinsic)] = extrinsic

        # camera
        camera = Pinhole(img_size=list(depth.shape[::-1]),
                         intrinsics=intrinsic[[0, 1, 0, 1], [0, 1, 2, 2]].tolist())
        frames.append(FrameWithConfidence(
            camera=camera, color=color, depth=depth, Tcw=np.linalg.inv(Twc), color_encoding="RGB",
            conf=conf, **FRAME_KWARGS, metadata={"gaussians": pred.gaussians}
        ))
    return frames


@dataclass
class DepthAnythingV3:
    size: str = "g"

    def __post_init__(self):
        model = {
            "s": "DA3-SMALL", "b": "DA3-BASE", "l": "DA3-LARGE-1.1", "g": "DA3-GIANT-1.1",
            "nested": "DA3NESTED-GIANT-LARGE-1.1", "mono": "DA3MONO-LARGE"
        }[self.size]

        repo_id = utils.huggingface_model_path(f"depth-anything/{model}")
        self.model = DepthAnything3.from_pretrained(repo_id).to(DEVICE)

    def inference(self, *args, **kwargs) -> list[FrameWithConfidence]:
        return prediction2frames(self.model.inference(*args, **kwargs))

    def mapping(self,
                frames: list[Frame],
                align_mode: str = "first",
                use_extrinsics: bool = False,
                **kwargs) -> SceneAligner:
        # predict
        if len(frames) == 1:
            pred = [self.monocular_depth(frames[0])]
        else:
            kwargs.update(dict(
                image=[f.cvt_color("RGB") for f in frames],
                extrinsics=np.stack([f.Twc for f in frames]) if use_extrinsics else None,
                intrinsics=np.stack([f.camera.camera_matrix for f in frames])
            ))
            pred = self.inference(**kwargs)

        # align
        sa = SceneAligner(frames_abs=frames, frames_rel=pred)
        sa.resize_rel_frames()
        sa.align_frames(mode=align_mode)
        sa.copy_metadata()
        return sa

    def monocular_depth(self,
                        frame: Frame) -> FrameWithConfidence:
        if self.size != "mono": warnings.warn("poor monocular depth performance")
        mask = frame.depth_mask.copy()
        frame = FrameWithConfidence(**(frame.__dict__ | FRAME_KWARGS), conf=np.full(mask.shape, 5.))
        pred = self.model.inference([frame.cvt_color("RGB")], intrinsics=frame.camera.camera_matrix[None])

        # conf
        if pred.sky is not None:
            frame.conf[resize_mask(pred.sky[0], frame.camera.img_size)] = 1.

        # resize, rectify
        depth = cv2.resize(pred.depth[0], frame.camera.img_size)
        frame.depth = align_depth(
            depth, frame.depth, mask=mask & frame.depth_mask)

        frame.sensor_depth = False
        return frame


class DA3Mapper:

    def __init__(self,
                 size: str = "g",
                 enable_mono: bool = True,
                 align_mode: str = "last"):
        self.main = DepthAnythingV3(size=size) if size else None
        self.mono = DepthAnythingV3(size="mono") if enable_mono else None
        self.align_mode = align_mode

    def __call__(self, frames: list[Frame]) -> SceneAligner:
        model = (self.mono or self.main) if len(frames) == 1 else self.main
        return model.mapping(frames, align_mode=self.align_mode)
