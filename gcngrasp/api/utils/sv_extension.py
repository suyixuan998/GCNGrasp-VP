from __future__ import annotations

import math
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import supervision as sv


def annotate_dets(image: np.ndarray,
                  dets: sv.Detections,
                  mask_opacity: float = 0.7,
                  anno_box: bool = True,
                  smart_label: bool = True) -> np.ndarray:
    has_class = dets.class_id is not None
    color_lookup = sv.ColorLookup.TRACK if dets.tracker_id is not None else (
        sv.ColorLookup.CLASS if dets.mask is None and has_class else sv.ColorLookup.INDEX)
    image = image.copy()

    if anno_box:
        anno_box = sv.BoxAnnotator(color_lookup=color_lookup)
        image = anno_box.annotate(image, detections=dets)

    if dets.mask is not None:
        anno_mask = sv.MaskAnnotator(color_lookup=color_lookup, opacity=mask_opacity)
        image = anno_mask.annotate(image, detections=dets)

    if dets.confidence is not None:
        anno_label = sv.LabelAnnotator(color_lookup=color_lookup, smart_position=smart_label)
        labels = [f"{score:.2f}" for score in dets.confidence]
        # class_name
        cls_name = get_class_name(dets)
        if cls_name:
            labels = [" ".join(fields) for fields in zip(cls_name, labels)]
        image = anno_label.annotate(image, detections=dets, labels=labels)

    return image


def annotate_kps(image: np.ndarray,
                 kps: sv.KeyPoints,
                 radius: float = 4,
                 cmap_cossim: str = "rainbow"):
    image = image.copy()

    desc = kps.data.get("descriptor")
    if desc is not None:
        score = (desc * desc[:1]).sum(axis=-1)
        color = plt.get_cmap(cmap_cossim)(score)[..., 2::-1] * 255
        xy = kps.xy.astype(int)
        for i in range(len(color)):
            for j in range(len(color[i])):
                cv2.circle(image, xy[i, j], radius, color[i, j], -1)

    else:
        anno_vex = sv.VertexAnnotator(radius=radius)
        image = anno_vex.annotate(image, kps)

    return image


def annotate_matches(img1: np.ndarray,
                     img2: np.ndarray,
                     kps1: sv.KeyPoints,
                     kps2: sv.KeyPoints,
                     matches: list[cv2.DMatch]) -> np.ndarray:
    process_kps = lambda kps: sum(kps_sv2cv(kps), [])
    return cv2.drawMatches(img1, process_kps(kps1), img2, process_kps(kps2), matches, None)


def box3d_area(xyzxyz: np.ndarray) -> np.ndarray:
    """ :param xyzxyz: shape = (N, 6) """
    return np.prod(xyzxyz[..., 3:] - xyzxyz[..., :3], axis=-1)


def box3d_iou_batch(boxes1: np.ndarray,
                    boxes2: np.ndarray) -> np.ndarray:
    """ :param boxes1, boxes2: `xyzxyz` format, shape = (N, 6) """
    inter = np.concatenate([np.maximum(boxes1[..., :3], boxes2[..., :3]),
                            np.minimum(boxes1[..., 3:], boxes2[..., 3:])], axis=-1)
    inter_vol = box3d_area(inter).clip(min=0)
    return inter_vol / (box3d_area(boxes1) + box3d_area(boxes2) - inter_vol)


def crop_images(image: np.ndarray,
                dets: sv.Detections,
                bg_value: Any = None) -> list[np.ndarray]:
    patches = [sv.crop_image(image, xyxy).copy() for xyxy in dets.xyxy]
    if bg_value is not None:
        for i, mask in enumerate(dets.mask):
            patches[i][~sv.crop_image(mask, dets.xyxy[i])] = bg_value
    return patches


def crop_image_assign(image: np.ndarray,
                      patch: np.ndarray,
                      xyxy: np.ndarray):
    x_min, y_min, x_max, y_max = xyxy
    image[y_min: y_max, x_min: x_max] = patch


def get_class_name(dets: sv.Detections) -> list[str]:
    if dets.class_id is None: return
    cls_name = dets.metadata.get("class_name")
    if cls_name is None: return
    return [cls_name[i] for i in dets.class_id]


def geo_sem_similarity(dets1: sv.Detections,
                       dets2: sv.Detections,
                       cross: bool,
                       iou_thresh: float = 0.01):
    """ :return: geometric and semantic similarity matrix"""
    dets1_feat = dets1.data["semantic"]
    dets2_feat = dets2.data["semantic"]
    dets1_mask_area = dets1.area
    dets2_mask_area = dets2.area

    if cross:
        sim = np.zeros([len(dets1), len(dets2), 2], dtype=np.float32)
        for i in range(len(dets1)):
            # filter by bbox iou
            is_valid = sv.box_iou_batch(dets1.xyxy[i, None], dets2.xyxy)[0] > iou_thresh
            dets2_mask = dets2.mask[is_valid]
            if not dets2_mask.size: continue
            # geometric similarity
            sim[i, is_valid, 0] = (dets1.mask[i] & dets2_mask).sum(axis=(1, 2)) / np.minimum(dets1_mask_area[i], dets2_mask_area[is_valid])
            # semantic similarity
            sim[i, is_valid, 1] = dets2_feat[is_valid] @ dets1_feat[i]

    else:
        assert len(dets1) == len(dets2), "dets1 and dets2 must have the same length for cross=False"
        sim = np.zeros([len(dets1), 2], dtype=np.float32)
        # geometric similarity
        sim[:, 0] = (dets1.mask & dets2.mask).sum(axis=(1, 2)) / np.minimum(dets1_mask_area, dets2_mask_area)
        # semantic similarity
        sim[:, 1] = (dets1_feat * dets2_feat).sum(axis=-1)

    return sim


def kernel_select(array, k, s=1, d=1, pad_value=0, axis=0):
    """
    See: torch.nn.functional.unfold
    :param array: NumPy array to be processed
    :param k: kernel size
    :param s: stride
    :param d: dilation
    :param pad_value: padding value
    :param axis: position of the kernel in the array
    """
    assert k & 1, "The size of the kernel should be odd"
    # Get the kernel coordinates
    coord = np.arange(- (k // 2), k // 2 + 1)
    kernel = np.stack(tuple(map(lambda x: x.T, np.meshgrid(coord, coord)[::-1])), axis=-1)
    kernel = kernel.reshape([-1, 2]) * d
    pad_width = kernel[0]
    # Pad the array to ensure the kernel can be applied correctly
    h, w = array.shape[:2]
    array_pad = np.pad(array, constant_values=pad_value,
                       pad_width=np.array(list(-pad_width) + [0] * (array.ndim - 2))[:, None].repeat(2, -1))
    return np.stack([array_pad[y:y + h:s, x:x + w:s] for x, y in kernel - pad_width], axis=axis)


def kps_sv2cv(kps: sv.KeyPoints) -> list[list[cv2.KeyPoint]]:
    n, m = kps.xy.shape[:2]
    class_id = kps.class_id
    response = kps.confidence
    angle = kps.data.get("angle")
    octave = kps.data.get("octave")
    size = kps.data.get("size")
    return [
        [cv2.KeyPoint(
            *kps.xy[i, j],
            size=size[i, j] if size is not None else 1,
            angle=angle[i, j] if angle is not None else -1.,
            response=response[i, j] if response is not None else 0.,
            octave=octave[i, j] if octave is not None else 0,
            class_id=class_id[i, j] if class_id is not None else -1
        ) for j in range(m)] for i in range(n)
    ]


def match_kps(kps1: sv.KeyPoints,
              kps2: sv.KeyPoints,
              dist_thresh: float = 0.15,
              dist_type: str = "cossim") -> list[cv2.DMatch]:
    assert kps1.xy.shape[1] == kps2.xy.shape[1] == 1
    desc1, desc2 = kps1.data["descriptor"][:, 0], kps2.data["descriptor"][:, 0]

    if dist_type == "cossim":
        distance = .5 - desc1 @ desc2.T / 2
    else:
        raise NotImplementedError(f"Unsupported distance type: {dist_type}")

    idx, matches = scipy.optimize.linear_sum_assignment(distance)
    dist = distance[idx, matches]
    mask = dist < dist_thresh
    return [cv2.DMatch(idx[i], matches[i], 0, dist[i]) for i in np.where(mask)[0]]


def refine_masks_by_ns(dets: sv.Detections,
                       pointmap: np.ndarray,
                       ksize: int = 3,
                       radius: float = .02,
                       nb_neighbors: int = 7) -> sv.Detections:
    assert 1 <= nb_neighbors < ksize ** 2, "nb_neighbors must be in [1, ksize^2)"
    pointmap = pointmap.copy()
    pointmap[pointmap[..., 2] <= 0] = np.inf

    for i in range(len(dets)):
        mask_ = sv.crop_image(dets.mask[i], dets.xyxy[i])
        pointmap_ = sv.crop_image(pointmap, dets.xyxy[i]).copy()
        pointmap_[~mask_] = np.inf

        # Select the 9 nearest neighbors for each point
        _9nn = kernel_select(pointmap_, ksize, pad_value=np.inf)[:, mask_]
        _center = _9nn.shape[0] // 2
        _not_ctr = np.arange(_9nn.shape[0]) != _center

        # Ignore invalid depth and padded points before subtracting (inf - inf).
        neighbors, center = _9nn[_not_ctr], _9nn[_center]
        valid = np.isfinite(neighbors).all(axis=-1) & np.isfinite(center).all(axis=-1)
        delta = np.zeros_like(neighbors)
        np.subtract(neighbors, center, out=delta, where=valid[..., None])
        nn = (valid & (np.linalg.norm(delta, axis=-1) <= radius)).sum(axis=0)

        # Apply the clustering result to the mask
        mask_new = np.zeros_like(mask_)
        mask_new[mask_] = nn >= nb_neighbors
        crop_image_assign(dets.mask[i], mask_new, dets.xyxy[i])

    return dets


def refine_masks_by_connectivity(dets: sv.Detections,
                                 min_region_area: int,
                                 connectivity: int) -> sv.Detections:
    valid_id = []
    for i in range(len(dets)):
        mask = sv.crop_image(dets.mask[i], dets.xyxy[i])
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)
        stats[0, -1] = 0  # set background area to 0

        # filter out small regions
        label_id = np.where(stats[:, -1] >= min_region_area)[0]
        if len(label_id):
            valid_id.append(i)
            if len(label_id) < num_labels - 1:
                mask &= np.isin(labels, label_id)
    return dets[valid_id]


def resize_mask(mask: np.ndarray,
                img_size: tuple[int, int]):
    return cv2.resize(mask.astype(np.uint8), img_size).astype(bool)


def plot_images_grid(images: list[np.ndarray],
                     titles: list[str] = None,
                     size: tuple[int, int] = (12, 12),
                     cmap: str = "gray"):
    n = len(images)
    w = math.ceil(math.sqrt(n))
    h = math.ceil(n / w)

    fig, axes = plt.subplots(nrows=h, ncols=w, figsize=size)
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            if images[idx].ndim == 2:
                ax.imshow(images[idx], cmap=cmap)
            else:
                ax.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))

            if titles is not None and idx < len(titles):
                ax.set_title(titles[idx])
        ax.axis("off")


if __name__ == '__main__':
    imgs = np.random.random([5, 100, 100])
    plot_images_grid(imgs)
    plt.show()
