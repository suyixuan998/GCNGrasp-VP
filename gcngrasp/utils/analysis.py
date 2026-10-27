from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import torch

from ..dataset import task_grasp
from ..api.utils import o3d_extension as o3de
from ..api.utils import skmetrics


def draw_error_prediction(obj: task_grasp.TaskGraspObject,
                          grasp_ids: np.ndarray,
                          y_true: np.ndarray,
                          y_score: np.ndarray,
                          cmap: str):
    thresh = skmetrics.bin_cls_optimal_thresh(y_true, y_score)[0]
    y_score = np.clip(y_score / thresh / 2, 0, 1)
    mask = (y_score > .5) != y_true
    return draw_prediction(obj, grasp_ids[mask], y_score[mask], cmap)


def draw_prediction(obj: task_grasp.TaskGraspObject,
                    grasp_ids: np.ndarray,
                    score: np.ndarray,
                    cmap: str = "coolwarm"):
    score = (score - score.min()) / (score.max() - score.min())
    score_ = np.zeros(len(obj), dtype=np.float32)
    for g, s in zip(grasp_ids, score):
        score_[g] = s
    geos = obj.mesh_grasp(score_, cmap=cmap)
    return [x for i, x in enumerate(geos) if i in grasp_ids]


def dump_prediction(batch: dict,
                    score: torch.Tensor) -> torch.Tensor:
    # obj_id, task_id, grasp_id, label, score, mask
    return torch.cat([
        batch["ot_id"][:, None].float().repeat(1, batch["grasp"].shape[1], 1),
        torch.stack([
            batch["grasp_id"].float(), batch["label"].float(), score, batch["mask"].float()
        ], dim=-1)
    ], dim=-1).flatten(start_dim=0, end_dim=1)


def load_prediction(file: str) -> pd.DataFrame:
    ret = pd.DataFrame(np.loadtxt(file), columns=list("otgls"))
    ret[list("otgl")] = ret[list("otgl")].astype(int)
    return ret


@dataclass
class TaskGraspPrediction:
    file: str
    dataset: task_grasp.TaskGrasp = None

    @cached_property
    def all_preds(self):
        return load_prediction(self.file)

    def ap_matrix(self,
                  thresh_ap: float = 0.,
                  cmap: str = "viridis_r",
                  bg_color: str = "white",
                  show: bool = True) -> np.ndarray:
        oids = np.sort(np.unique(self.all_preds["o"]))
        shape = len(self.dataset.tasks), max(oids)

        aps = np.full(shape, -1, dtype=np.float32)
        for (o, t), v in skmetrics.mAP_score(
                self.all_preds[list("ot")].to_numpy(), self.all_preds["l"], self.all_preds["s"], ret_org=True).items():
            if v < thresh_ap: continue
            aps[t, o - 1] = v

        mask = aps >= 0
        img = np.broadcast_to(mcolors.to_rgb(bg_color), shape + (3,)).copy()
        img[mask] = plt.get_cmap(cmap)(aps[mask])[..., :3]
        img = img[:, oids - 1]

        if show:
            plt.xlabel("Object ID")
            plt.ylabel("Task ID")
            plt.xticks(np.arange(img.shape[1]), oids, rotation=90)
            plt.yticks(np.arange(shape[0]), self.dataset.tasks)
            plt.imshow(img)
            plt.show()
        return img

    @cached_property
    def gold_preds(self):
        return self.all_preds[self.dataset.is_gold_ot(self.all_preds[list("ot")].to_numpy())]

    def visualize_error_pred(self,
                             save_dir: Path,
                             cmap: str = "coolwarm"):
        save_dir.mkdir(parents=True, exist_ok=True)

        for ot, gls in self.gold_preds.groupby(by=list("ot")):
            o: task_grasp.TaskGraspObject = self.dataset.load_obj(ot[0])
            gls = gls[list("gls")].to_numpy()

            geos = draw_error_prediction(o, *gls.T, cmap=cmap)
            if not geos: continue

            img = o3de.auto_render_color([640, 480], geos)
            index = f"{o.name}-{self.dataset.tasks[ot[1]]}"
            cv2.imwrite(str(save_dir / f"{index}-error.png"), img)
