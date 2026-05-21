import warnings

import numpy as np
import pytorch_lightning as pl
import torch

from ..data_type.field import BaseIGField
from ... import ConfigCollector
from ...api.utils import o3d_extension as o3de


class TOGEvaluatorAndAffordance(pl.LightningModule):

    def __init__(self,
                 config: dict):
        super().__init__()
        self.config = config
        self.cfg_col = ConfigCollector.from_path(config["path"])
        self.n_limit = [config["n_points_min"], self.cfg_col.data_config["n_points"]]

        self.model = self.cfg_col.model_type(self.cfg_col.model_config).to(config["device"]).eval()
        self.load_state_dict(torch.load(self.cfg_col.ckpt_best, weights_only=True, map_location=self.model.device)["state_dict"])

    @torch.inference_mode()
    def forward(self,
                pcd: np.ndarray,
                grasp: np.ndarray,
                obj: str,
                task: str
                ) -> tuple[np.ndarray, BaseIGField]:
        n = max(self.n_limit[0], min(len(pcd) // 2, self.n_limit[1]))
        pcd = pcd[o3de.regularize_pcd_size(pcd, n, use_fps=True)]

        device = self.model.device
        with torch.inference_mode():
            ret = self.model(
                torch.from_numpy(pcd)[None].to(device), torch.from_numpy(grasp)[None].to(device), [obj], [task],
                test_time_augment=self.config.get("tta", 0)
            )

        aff = ret.get("affordance")
        if aff:
            field = BaseIGField(pcd=aff["xyz"][0].cpu().numpy(), ig=aff["score"][0].cpu().numpy())
            field = field.interp(pcd)
        else:
            field = None
        return ret["score"][0].cpu().numpy(), field
