import numpy as np
import supervision as sv

from .grounding_dino import GroundingDINO
from .segment_anything import SegmentAnythingV2


class GroundedSAM:

    def __init__(self,
                 gdino_kwargs: dict = {},
                 sam_kwargs: dict = {}):
        self.gdino = GroundingDINO(**gdino_kwargs)
        self.sam = SegmentAnythingV2(**sam_kwargs)

    def __call__(self,
                 bgr: np.ndarray,
                 obj: str,
                 topk: int = None) -> sv.Detections:
        dets = self.gdino(bgr, obj)
        if topk: dets = dets[:topk]
        dets = self.sam.inference_with_box(bgr, dets)
        return dets
