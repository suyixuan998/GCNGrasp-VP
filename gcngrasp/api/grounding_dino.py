from dataclasses import dataclass

import numpy as np
import supervision as sv
from groundingdino.util import inference
from pathlib import Path

from . import utils


@dataclass
class GroundingDINO:
    """ Reference: https://github.com/IDEA-Research/Grounded-Segment-Anything/blob/main/automatic_label_tag2text_demo.py"""
    encoder: str = "SwinB_cogcoor"
    box_thresh: float = 0.35
    text_thresh: float = 0.25
    nms_iou: float = 0.5
    boxarea_thresh: float = 0.7

    def __post_init__(self):
        # from groundingdino.config import GroundingDINO_SwinB_cfg
        config_dir = Path(__file__).parent / "third_party/config"
        weight_file = utils.download_checkpoint(
            f"https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_{self.encoder.lower()}.pth")
        self.model = inference.Model(config_dir / f"grounding_dino/GroundingDINO_{self.encoder}.py", weight_file)

    def __call__(self,
                 image: np.ndarray,
                 caption: str) -> sv.Detections:
        """
        Open-vocabulary object detection
        :param image: OpenCV image
        :param caption: text prompt
        """
        dets, phrases = self.model.predict_with_caption(
            image, caption,
            box_threshold=self.box_thresh,
            text_threshold=self.text_thresh
        )
        dets.metadata["class_name"] = classes = np.array(sorted(set(phrases)))
        dets.class_id = np.searchsorted(classes, phrases)
        if self.boxarea_thresh:
            dets = dets[dets.box_area / np.prod(image.shape[:2]) < self.boxarea_thresh]
        if self.nms_iou: dets = dets.with_nms(self.nms_iou)
        return dets
