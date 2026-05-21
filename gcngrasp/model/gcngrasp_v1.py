# https://github.com/adithyamurali/TaskGrasp
import re
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
import torch_geometric.utils
from torch import nn

from . import backbone as bb
from . import common
from . import pointnet2_utils as pnu
from . import utils


class GraphNet(nn.Module):
    """Class for Graph Convolutional Network"""

    def __init__(self, c1, c2, e, n, conv_type, drop):
        super().__init__()
        c_ = round(c2 * e)
        ConvLayer = getattr(gnn, conv_type)
        self.drop = drop
        self.c2 = c2

        self.convs = nn.ModuleList([ConvLayer(c1, c_)])
        self.batch_norms = nn.ModuleList([gnn.BatchNorm(c_)])
        for _ in range(n - 2):
            self.convs.append(ConvLayer(c_, c_))
            self.batch_norms.append(gnn.BatchNorm(c_))
        self.convs.append(ConvLayer(c_, c2))

    def forward(self, x, edge_index):
        for conv, batch_norm in zip(self.convs[:-1], self.batch_norms):
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.drop, training=self.training)
        return self.convs[-1](x, edge_index)


class GraphReadout(common.KnowledgeGraph):

    def __init__(self, config: dict):
        super().__init__(config["c1"], config)
        self.gcn = GraphNet(self.c2, **{k: config[k] for k in ("c2", "e", "n", "conv_type", "drop")})
        self.c2 = config["c2"]

    def forward(self,
                obj_embedding: torch.Tensor,
                objs: List[str],
                tasks: List[str]):
        """
        :param obj_embedding: object embedding [B, C]
        :param objs: object name [B]
        :param tasks: task name [B]
        :return: obj_embedding [B, C']
        """
        assert obj_embedding.shape[0] == len(objs) == len(tasks)
        graphs = torch_geometric.data.Batch.from_data_list([self.process_input(*args) for args in zip(obj_embedding, objs, tasks)])
        outs = self.gcn(graphs.x, graphs.edge_index)
        return outs[graphs.ptr[1:] - 1]


class GCNGraspV1(pl.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.graph = GraphReadout(config["graph"])

        self.SA_modules = bb.PointNet2(c1=4, **bb.PointNet2.MSG_CLS)

        c = config["fc1"]
        self.fc_layer = nn.Sequential(
            nn.Linear(*c[:2], bias=False), nn.BatchNorm1d(c[1]), nn.ReLU(True),
            nn.Linear(c[1], config["graph"]["c1"])
        )

        c1 = self.graph.c2
        self.fc_layer3 = nn.Sequential()
        for i in range(3):
            self.fc_layer3.extend((
                nn.Linear(c1, c1, bias=i != 0), nn.BatchNorm1d(c1), nn.ReLU(True), nn.Dropout(config["drop"]),
            ))
        self.fc_layer3.append(nn.Linear(c1, 1))

    def example_input(self, B, S, N):
        dev_dtype = self.graph.embedding.weight
        return torch.rand(B, N, 3).to(dev_dtype), torch.rand(B, S, 6, 3).to(dev_dtype), *self.graph.example_input(B)[1:]

    def forward(self,
                pcd: torch.Tensor,
                grasp: torch.Tensor,
                objs: list[str],
                tasks: list[str]) -> dict:
        """
        :param pcd: colored point cloud of the object [B, N, 3]
        :param grasp: gripper control points [B, S, G, 3]
        :param objs: object name [B]
        :param tasks: task name [B]
        """
        B, S, G, _ = grasp.shape
        grasp = grasp[:, :, [0, 1, 2, 4, 2, 3, 5]]
        objs = sum([[o] * S for o in objs], [])
        tasks = sum([[t] * S for t in tasks], [])

        # [BS, N + 6, 4]
        latent_pcd = self.process_pcd_grasp(*utils.normalize_pcd(pcd, grasp))
        latent_pcd = self.SA_modules(latent_pcd)

        # [BS, C]
        x = self.fc_layer(latent_pcd.feature.squeeze(-2))
        x = self.graph(x, objs, tasks)
        x = self.fc_layer3(x).view(B, S)
        return {"logits": x, "score": torch.sigmoid(x)}

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

    def visualize_for_debug(self,
                            grasp: torch.Tensor,
                            pcd: torch.Tensor):
        from .utils import visualize_grasp
        visualize_grasp(grasp.flatten(0, 1), pcd.flatten(0, 1))

    def load_org_ckpt(self, file: str):
        state_dict = torch.load(file, weights_only=True, map_location="cpu")["state_dict"]
        self.load_state_dict(self.align_state_dict(state_dict), strict=True)
        print(f'Loaded model from "{file}"')

    @staticmethod
    def align_state_dict(state_dict: dict) -> dict:
        state_dict = pnu.align_pointnet2_state_dict(state_dict)
        for k in list(state_dict):
            knew = k
            if re.search(r"gcn\.convs\..+\.weight", k):
                knew = k.replace("weight", "lin.weight")
                state_dict[k] = state_dict[k].T
            elif re.search(r"gcn\.batch_norms", k):
                knew = ".module.".join(k.rsplit(".", 1))
            elif k.split(".")[0] in ("task_embedding", "class_embedding"):
                knew = ""
            elif "graph_embedding" in k:
                knew = k.replace("graph_embedding", "graph.embedding")
            knew = knew.replace("gcn.", "graph.gcn.")
            if not knew:
                state_dict.pop(k)
            elif knew != k:
                state_dict[knew] = state_dict.pop(k)
        return state_dict
