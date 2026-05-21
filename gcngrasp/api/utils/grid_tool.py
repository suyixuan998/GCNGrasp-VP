from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv

ANNO_CONFIG = dict(
    color=(255, 255, 255),
    thickness=5e-3,
    thickness_min=1,
    fontscale=2e-3
)


@dataclass
class GridLine:
    x: np.ndarray  # x-coordinates of the grid lines
    y: np.ndarray  # y-coordinates of the grid lines

    def __post_init__(self):
        self.gh = len(self.y) - 1
        self.gw = len(self.x) - 1

    def __getitem__(self, item):
        """ Get the bounding box of the grid cell by its ID. """
        assert item >= 0
        r, c = item // (self.x.size - 1), item % (self.x.size - 1)
        return np.array([self.x[c], self.y[r], self.x[c + 1], self.y[r + 1]])

    def to_detections(self,
                      indices: list[int] = None) -> sv.Detections:
        """ Convert the grid lines to Detections object. """
        indices = indices or list(range(self.gh * self.gw))
        return sv.Detections(
            xyxy=np.stack([self[i] for i in indices]),
            tracker_id=np.array(indices, dtype=int),
        )

    def annotate(self,
                 image: np.ndarray,
                 indices: list[int] = None,
                 id_fmt: str = "%d"):
        """ Annotate the image with a uniform grid and grid IDs."""
        image = image.copy()
        h, w = image.shape[:2]
        dets = self.to_detections(indices)
        # draw grid lines
        thickness = max(ANNO_CONFIG["thickness_min"], round(min(h, w) * ANNO_CONFIG["thickness"]))
        anno_box = sv.BoxAnnotator(color=sv.Color.from_bgr_tuple(ANNO_CONFIG["color"]),
                                   color_lookup=sv.ColorLookup.TRACK, thickness=thickness)
        anno_box.annotate(image, detections=dets)
        # grid ID
        fontscale = min(h, w) * ANNO_CONFIG["fontscale"]
        for i, xyxy in zip(dets.tracker_id, dets.xyxy):
            offset = thickness * 2
            cv2.putText(image, id_fmt % i, (xyxy[0] + offset, xyxy[-1] - offset),
                        cv2.FONT_HERSHEY_SIMPLEX, fontscale, ANNO_CONFIG["color"], thickness)
        return image


@dataclass
class GridPoint:
    x: np.ndarray
    y: np.ndarray

    def array(self, flatten=False):
        ret = np.stack(np.meshgrid(self.x, self.y), axis=-1)
        return ret.reshape(-1, 2) if flatten else ret


class UniformGrid:

    def __init__(self,
                 size: int):
        self.size = size

    def cal_shape(self, w, h):
        return np.round(np.sqrt([self.size * h / w, self.size / h * w])).astype(int)

    def generate_point(self, w, h):
        nr, nc = self.cal_shape(w, h)
        gsr, gsc = h / nr, w / nc
        rows = np.linspace(gsr, h - gsr, nr)
        cols = np.linspace(gsc, w - gsc, nc)
        return GridPoint(x=cols, y=rows)

    def generate_grid(self, w, h):
        """ Generate grid information based on the image size. """
        nr, nc = self.cal_shape(w, h)
        rows = np.round(np.linspace(0, h - 1, nr + 1)).astype(int)
        cols = np.round(np.linspace(0, w - 1, nc + 1)).astype(int)
        return GridLine(x=cols, y=rows)


if __name__ == '__main__':
    ug = UniformGrid(30)
    grid = ug.generate_grid(640, 480)

    img = cv2.imread(r"D:\Workbench\Lab\Unnamed\assets\green-pot\color\000.png")
    img = grid.annotate(img, indices=[1, 3, 6, 10])
    cv2.imshow("grid", img)
    cv2.waitKey(0)
