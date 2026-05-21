from __future__ import annotations

import concurrent.futures
import threading

import cv2
import numpy as np
import supervision as sv
import torch
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from . import utils
from .utils import grid_tool

DEVICE, DTYPE = utils.get_tensor_args()


def drop_duplicated(dets_sam2: sv.Detections,
                    max_overlap_radio: float = 0.9):
    # Sort detections by area
    area = dets_sam2.area
    _i = np.argsort(area)
    dets_sam2, area = dets_sam2[_i], area[_i]
    # Filter out duplicated detections
    is_valid = np.ones(len(dets_sam2), dtype=bool)
    iou = sv.box_iou_batch(dets_sam2.xyxy, dets_sam2.xyxy)
    for i in range(len(dets_sam2)):
        for j in range(i + 1, len(dets_sam2)):
            if iou[i, j] == 0: continue
            inter = (dets_sam2.mask[i] & dets_sam2.mask[j]).sum()
            if inter / min(area[[i, j]]) > max_overlap_radio:
                is_valid[i] = False
                break
    return dets_sam2[is_valid]


def to_detections(result) -> sv.Detections:
    if isinstance(result, tuple):
        masks, scores, _ = result
        if masks.ndim > 3:
            masks = masks.reshape(-1, *masks.shape[-2:])
            scores = scores.reshape(-1)
        return sv.Detections(xyxy=sv.mask_to_xyxy(masks), mask=masks.astype(np.bool_), confidence=scores)

    elif isinstance(result, list):
        return sv.Detections(
            xyxy=sv.xywh_to_xyxy(np.array([r["bbox"] for r in result], dtype=int)),
            mask=np.stack([r["segmentation"] for r in result]),
            confidence=np.array([r["predicted_iou"] for r in result])
        ) if result else sv.Detections.empty()

    else:
        raise TypeError(f"Unsupported type for result: {type(result)}")


class ContinousPredictor:

    def __init__(self,
                 predictor,
                 image: np.ndarray):
        if isinstance(predictor, SAM2ImagePredictor):
            self.img_pred = predictor
            self.img_pred.set_image(image)

        elif isinstance(predictor, SAM2AutomaticMaskGenerator):
            self.img_pred = predictor.predictor
            # Reset-timing: set_image(), generate()
            self.img_pred.reset_predictor = lambda: None  # print("Skip reset predictor.")
            self.__gen = to_detections(predictor.generate(image))

        else:
            raise TypeError(f"Unsupported predictor type: {type(predictor)}")

    @torch.autocast(DEVICE, dtype=DTYPE)
    def __call__(self,
                 **kwargs) -> sv.Detections:
        return to_detections(self.img_pred.predict(
            **kwargs, return_logits=False, multimask_output=False)) if kwargs else self.__gen


class SegmentAnythingV2:
    lock = threading.Lock()

    def __init__(self,
                 encoder: str,
                 img_pred_kwargs: dict = None,
                 auto_mask_kwargs: dict = None):
        """
        https://github.com/facebookresearch/sam2
        :param encoder: Encoder type (tiny, small, base_plus, large)
        """
        encoder_type = {"tiny": "t", "small": "s", "base_plus": "b+", "large": "l"}
        assert encoder in encoder_type

        checkpoint = f"checkpoints/sam2.1_hiera_{encoder}.pt"
        model_cfg = f"configs/sam2.1/sam2.1_hiera_{encoder_type[encoder]}.yaml"
        self.model = build_sam2(model_cfg, checkpoint)

        self.img_pred_kwargs: dict = img_pred_kwargs or {}
        self.img_pred_getter = lambda: SAM2ImagePredictor(self.model, **self.img_pred_kwargs)
        # self.video_predictor = build_sam2_video_predictor(model_cfg, checkpoint)

        self.auto_mask_kwargs: dict = auto_mask_kwargs or {}
        self.uni_grid = grid_tool.UniformGrid(self.auto_mask_kwargs.pop("points_per_side", 32) ** 2)
        self.auto_mask_getter = lambda img_size: SAM2AutomaticMaskGenerator(
            self.model, points_per_side=None, crop_n_layers=0,
            point_grids=[self.uni_grid.generate_point(*img_size).array(flatten=True) / img_size],
            **self.auto_mask_kwargs
        )

    def build_predictor(self,
                        image: np.ndarray,
                        auto_mode: bool) -> ContinousPredictor:
        """ Build a predictor for continuous inference """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with self.lock:
            predictor = self.auto_mask_getter(image.shape[1::-1]) if auto_mode else self.img_pred_getter()
        return ContinousPredictor(predictor, image)

    def inference_with_box(self,
                           image: np.ndarray,
                           dets: sv.Detections | concurrent.futures.Future[sv.Detections]) -> sv.Detections:
        predictor = self.build_predictor(image, auto_mode=False)
        if not isinstance(dets, sv.Detections): dets = dets.result()
        return predictor(box=dets.xyxy) if dets else sv.Detections.empty()

    @torch.autocast(DEVICE, dtype=DTYPE)
    def __call__(self,
                 image: np.ndarray,
                 **kwargs) -> sv.Detections:
        """ :param kwargs: use `prompt mode` if keyword parameters are provided, otherwise use `automatic mode` """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if kwargs:
            with self.lock:
                img_pred = self.img_pred_getter()
            img_pred.set_image(image)
            result = img_pred.predict(**kwargs, return_logits=False, multimask_output=False)
        else:
            with self.lock:
                auto_mask = self.auto_mask_getter(image.shape[1::-1])
            result = auto_mask.generate(image)
        return to_detections(result)


SegmentAnythingV2.__call__.__doc__ = ContinousPredictor.__call__.__doc__ = SAM2ImagePredictor.predict.__doc__
