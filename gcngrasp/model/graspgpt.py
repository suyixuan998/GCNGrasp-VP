from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn

from . import backbone as bb
from . import pointnet2_utils as pnu
from . import utils


class GraspGPT(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.SA_modules = bb.PointNet2(c1=4, **bb.PointNet2.MSG_CLS)

        c_e = config["embedding"]
        self.obj_preprocess = nn.Linear(*c_e)
        self.task_preprocess = nn.Linear(*c_e)
        self.ins_preprocess = nn.Linear(*c_e)

        c_fc = config["fc1"]
        self.fc_layer = nn.Sequential(
            nn.Linear(*c_fc[:2], bias=False), nn.BatchNorm1d(c_fc[1]), nn.ReLU(True),
            nn.Linear(*c_fc[1:3])
        )

        c2 = c_e[-1]
        c1 = c_fc[-1] + c2 * 3
        self.fc_layer3 = nn.Sequential()
        for i in range(3):
            self.fc_layer3.extend((
                nn.Linear(c1, c2, bias=i != 0), nn.BatchNorm1d(c2), nn.ReLU(True), nn.Dropout(config["drop"]),
            ))
            c1 = c2
        self.fc_layer3.append(nn.Linear(c1, 1))

    def example_input(self, B, S, N, L=128):
        dev_dtype = self.ins_preprocess.weight
        features = torch.rand(3, B, L, self.ins_preprocess.in_features).to(dev_dtype)
        masks = torch.ones([3, B, L], dtype=torch.bool, device=dev_dtype.device)
        descs = [{"feature": features[i], "mask": masks[i]} for i in range(len(features))]
        return torch.rand(B, N, 3).to(dev_dtype), torch.rand(B, S, 6, 3).to(dev_dtype), *descs

    def forward(self,
                pcd: torch.Tensor,
                grasp: torch.Tensor,
                objs: dict[str, torch.Tensor],
                tasks: dict[str, torch.Tensor],
                ins: dict[str, torch.Tensor]) -> dict:
        """
        :param pcd: colored point cloud of the object [B, N, 3]
        :param grasp: gripper control points [B, S, G, 3]
        """
        B, S, G, _ = grasp.shape
        grasp = grasp[:, :, [0, 1, 2, 4, 2, 3, 5]]

        # [BS, N + 6, 4]
        latent_pcd = self.process_pcd_grasp(*utils.normalize_pcd(pcd, grasp))
        latent_pcd = self.SA_modules(latent_pcd)
        x1 = self.fc_layer(latent_pcd.feature.squeeze(1))

        objs = self.forward_text_embedding(objs, self.obj_preprocess)
        tasks = self.forward_text_embedding(tasks, self.task_preprocess)
        ins = self.forward_text_embedding(ins, self.ins_preprocess)
        x2 = torch.cat([objs, tasks, ins], dim=1)
        x2 = x2[:, None].repeat(1, S, 1).flatten(start_dim=0, end_dim=1)

        x = torch.cat([x1, x2], dim=1)
        x = self.fc_layer3(x).view(B, S)
        return {"logits": x, "score": torch.sigmoid(x)}

    def forward_text_embedding(self,
                               text: dict[str, torch.Tensor],
                               proj: nn.Module):
        # [B, L, C] -> [B, C]
        feat = proj(text["feature"])
        mask = text["mask"].unsqueeze(-1).float()
        return torch.sum(feat * mask, dim=1) / torch.sum(mask, dim=1)

    def process_pcd_grasp(self,
                          pcd: torch.Tensor,
                          grasp: torch.Tensor) -> pnu.Latent:
        N = pcd.shape[1]
        B, S, G, _ = grasp.shape
        pcd = pcd[..., :3][:, None].repeat(1, S, 1, 1)
        # self.visualize_for_debug(grasp, pcd)

        # [B, N, 3] -> [B, S, N+G, 3] -> [BS, N+G, 3]
        xyz = torch.cat([pcd, grasp], dim=-2).flatten(start_dim=0, end_dim=1)
        flag = torch.cat([torch.zeros([B * S, N]), torch.ones([B * S, G])], dim=-1).unsqueeze(-1).to(pcd.device)
        return pnu.Latent(xyz, flag)

    def load_org_ckpt(self, file: str):
        state_dict = torch.load(file, weights_only=True, map_location="cpu")["state_dict"]
        self.load_state_dict(self.align_state_dict(state_dict), strict=True)
        print(f'Loaded model from "{file}"')

    @staticmethod
    def align_state_dict(state_dict: dict) -> dict:
        state_dict = pnu.align_pointnet2_state_dict(state_dict)
        return state_dict
