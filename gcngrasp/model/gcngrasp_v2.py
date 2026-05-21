import copy
from typing import List

import numpy as np
import pytorch_lightning as pl
import scipy
import torch
import torch_geometric.utils
from torch import nn

from . import backbone as bb
from . import common
from . import pointnet2_utils as pnu
from . import utils

SO3 = scipy.spatial.transform.Rotation


class GraphReadout(common.KnowledgeGraph):

    def __init__(self, c1, c2, e, n, drop,
                 config_graph: dict):
        super().__init__(c1, config_graph)
        c1 = self.c2
        c_ = round(c2 * e)

        self.gcn = common.GCNConvBnAct.create_mlp(c1, [c_] * (n - 2) + [c2])
        # m = self.gcn[-1]
        for m in self.gcn:
            m.act = nn.Sequential(m.act, nn.Dropout(drop))
        self.c2 = c2

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
        x = self.gcn(graphs).x
        return x[graphs.ptr[1:] - 1]


class GraspScorer(nn.Module):

    def __init__(self,
                 c1: int,
                 c2s: list[int],
                 c2s_head: list[int],
                 k: int,
                 drop: float,
                 gripper_info: dict):
        super().__init__()
        self.k = k
        self.gripper_info = gripper_info

        self.mlp = common.ConvBnAct2d.create_mlp(c1, c2s, k=1)
        c1 = self.mlp.c2 * gripper_info["n_pts"]

        self.head = common.ConvBnAct1d.create_mlp(c1, c2s_head, k=1)
        self.head.append(nn.Conv1d(c2s_head[-1], 1, kernel_size=1))
        self.head[0].act = nn.Sequential(nn.ReLU(), nn.Dropout(drop))

    def forward(self,
                grasp: torch.Tensor,
                latent_ot: pnu.Latent,
                random_aug: bool = False):
        B, G = grasp.shape[:2]
        bi = torch.repeat_interleave(torch.arange(B), G).to(latent_ot.feature.device)
        latent_ot = pnu.Latent(xyz=latent_ot.xyz[bi], feature=latent_ot.feature[bi])

        # transform to gripper frame
        grasp = grasp.flatten(start_dim=0, end_dim=1)
        grasp, latent_ot.xyz = self.transform_to_gripper(grasp, latent_ot.xyz, random_flip=random_aug or self.training)
        latent_grasp = pnu.Latent.from_tensor(grasp)
        # self.visualize_for_debug(grasp, latent_ot.xyz)

        # contact points feature
        latent_grasp = pnu.group_ops(latent_ot, latent_grasp, mlp=self.mlp, k=self.k,
                                     group_type=self.query_contact_point, concat_bias=False)
        x = latent_grasp.feature.flatten(start_dim=1).view(B, G, -1).permute(0, 2, 1)
        x = self.head(x).squeeze(1)
        return {"logits": x, "score": torch.sigmoid(x)}

    def query_contact_point(self,
                            pcd: torch.Tensor,
                            grasp: torch.Tensor,
                            k: int) -> torch.Tensor:
        g_min, g_max = grasp.min(dim=-2, keepdim=True).values, grasp.max(dim=-2, keepdim=True).values
        g_min[..., 1] = - torch.inf
        g_max[..., 1:] = torch.inf
        mask = torch.all((pcd >= g_min) & (pcd <= g_max), dim=-1)

        # filter out points outside the gripper
        pcd = pcd.clone()
        pcd[~mask] = torch.inf
        sqrdists = pnu.square_distance(grasp, pcd)

        # query nearest k points
        dist, group_idx = torch.topk(sqrdists, k=k, dim=-1, largest=False, sorted=False)
        group_first = group_idx[..., :1].repeat(1, 1, k)
        mask = ~torch.isfinite(dist)
        group_idx[mask] = group_first[mask]
        return group_idx

    def transform_to_gripper(self,
                             grasp: torch.Tensor,
                             *pcds,
                             random_flip: bool = False):
        z = torch.diff(grasp[:, self.gripper_info["axis_z"]], axis=1)[:, 0]
        x = torch.diff(grasp[:, self.gripper_info["axis_x"]], axis=1)[:, 0]
        y = torch.cross(z, x, dim=-1)
        x = torch.cross(y, z, dim=-1)
        R = torch.stack([x, y, z], dim=-1)
        R /= torch.norm(R, dim=-2, keepdim=True)

        B = grasp.shape[0]
        center = grasp[:, self.gripper_info["center"]][:, None]
        grasp = torch.bmm(grasp - center, R)

        if random_flip:
            flip = 2. * (torch.rand([B, 1, 2]) > .5) - 1
            R[..., :2] *= flip.to(grasp.device)

        pcds = [torch.bmm(xyz - center, R) for xyz in pcds]
        return grasp, *pcds

    def visualize_for_debug(self,
                            grasp: torch.Tensor,
                            pcd: torch.Tensor):
        from .utils import visualize_grasp
        visualize_grasp(grasp, pcd, group_pcd=self.query_contact_point(pcd, grasp, k=self.k),
                        connections=self.gripper_info["connections"])


class AffordanceHead(nn.Module):

    def __init__(self,
                 c1: int,
                 c2s: list[int],
                 c2s_head: list[int],
                 n2: int,
                 k: int,
                 r: float,
                 drop: float):
        super().__init__()
        self.sa = pnu.PointNetSetAbstractionMsg(c1, c2s, n2, k, r, drop=drop)
        self.head = common.ConvBnAct1d.create_mlp(self.sa.c2, c2s_head, k=1)
        self.head.append(nn.Conv1d(self.head.c2, 1, kernel_size=1))
        self.act = nn.Softmax(dim=-1)

    def forward(self,
                latent_ot: pnu.Latent):
        latent_aff = self.sa(latent_ot)
        x = self.head(latent_aff.feature.permute(0, 2, 1)).squeeze(1)
        return {"xyz": latent_aff.xyz, "logits": x, "score": self.act(x)}


class GCNGraspV2(pl.LightningModule):

    def __init__(self,
                 config: dict,
                 include: list[str] = None,
                 exclude: list[str] = None):
        super().__init__()
        config_input: dict = config["input"]
        config_model: dict = copy.deepcopy(config["model"])

        # modify config
        assert not all([include, exclude])
        if include:
            config_model = {k: v for k, v in config_model.items() if k in include}
        elif exclude:
            for k in exclude: config_model.pop(k)

        w = config["width_multiplier"]
        for name, cfg in config_model.items():

            # handle non-dict config
            if not isinstance(cfg, dict):
                if name == "pcd_encoder":
                    cfg = getattr(bb.PointNet2, cfg)

            # apply width_multiplier
            for attr in cfg:
                if attr.startswith("c2s"):
                    cfg[attr] = common.apply_width_multiplier(cfg[attr], w)

            # initialize
            if name == "pcd_encoder":
                m = nn.ModuleList(bb.PointNet2(c1=3, **cfg))
                n = len(m) - sum(map(lambda x: isinstance(x, pnu.PointNetFeaturePropagation), m))
                m = nn.ModuleList([nn.Sequential(*m[:n]), nn.Sequential(*m[n:])])
            elif name == "graph_readout":
                c1 = self.pcd_encoder[0][-1].c2
                m = GraphReadout(c1, c1, **cfg, config_graph=config_input["graph"])
            elif name == "grasp_scorer":
                m = GraspScorer(self.pcd_encoder[1][-1].c2, gripper_info=config_input["gripper"], **cfg)
            elif name == "affordance_head":
                m = AffordanceHead(self.pcd_encoder[1][-1].c2, **cfg)
            else:
                raise ValueError(f"Unknown layer name: {name}")
            setattr(self, name, m)

    def example_input(self, B, S, N):
        dev_dtype = self.graph_readout.embedding.weight
        return torch.rand(B, N, 3).to(dev_dtype), torch.rand(B, S, 6, 3).to(dev_dtype), *self.graph_readout.example_input(B)[1:]

    def forward_impl(self,
                     pcd: torch.Tensor,
                     grasp: torch.Tensor,
                     objs: list[str],
                     tasks: list[str],
                     random_aug: bool = False) -> dict:
        """
        :param pcd: colored point cloud of the object [B, N, 3]
        :param grasp: gripper control points [B, S, G, 3]
        :param objs: object name [B]
        :param tasks: task name [B]
        """
        normalizer = utils.PcdNormalizer(pcd, scaling=True)

        latent_pcd = pnu.Latent.from_tensor(normalizer.forward(pcd))
        latent_pcd = self.pcd_encoder[0](latent_pcd)

        # graph
        B, K, _ = latent_pcd.feature.shape
        objs = sum([[o] * K for o in objs], [])
        tasks = sum([[t] * K for t in tasks], [])
        x = self.graph_readout(latent_pcd.feature.flatten(start_dim=0, end_dim=1), objs, tasks)

        # pcd
        latent_pcd.feature = x.view(B, K, -1)
        latent_pcd = self.pcd_encoder[1](latent_pcd)
        ret = {}

        # grasp
        if grasp is not None:
            grasp = normalizer.forward(grasp)
            ret.update(self.grasp_scorer(grasp, latent_pcd, random_aug))

        # affordance
        aff_head = getattr(self, "affordance_head", None)
        if aff_head is not None:
            ret["affordance"] = aff_head(latent_pcd)
            ret["affordance"]["xyz"] = normalizer.backward(ret["affordance"]["xyz"])
        return ret

    def forward(self,
                pcd: torch.Tensor,
                grasp: torch.Tensor,
                objs: list[str],
                tasks: list[str],
                test_time_augment: int = 1):
        if test_time_augment <= 1: return self.forward_impl(pcd, grasp, objs, tasks)

        # repeat
        B, S = grasp.shape[:2]
        rotvec = np.random.uniform(0, np.pi * 2, [B * test_time_augment, 3])
        rotvec[:, :2] = 0
        R = torch.from_numpy(SO3.from_rotvec(rotvec).as_matrix()).to(pcd)
        pcd = pcd.repeat(test_time_augment, 1, 1) @ R
        grasp = grasp.repeat(test_time_augment, 1, 1, 1) @ R[:, None]
        objs = objs * test_time_augment
        tasks = tasks * test_time_augment

        # forward
        ret = self.forward_impl(pcd, grasp, objs, tasks, random_aug=True)
        ret.pop("logits")
        afford = ret.get("affordance")

        # average
        ret["score"] = ret["score"].view(test_time_augment, B, -1).mean(dim=0)
        if afford:
            xyz = (afford["xyz"] @ R.permute(0, 2, 1)).view(test_time_augment, B, -1, 3)
            logits = afford["logits"].view(*xyz.shape[:-1], 1)

            for i in range(B):
                idx = pnu.square_distance(xyz[:1, i], xyz[:, i]).argmin(dim=-1)
                logits[0, i] = pnu.index_points(logits[:, i], idx).mean(dim=0)

            afford["xyz"] = xyz[0]
            afford["logits"] = logits[0].squeeze(-1)
            afford["score"] = self.affordance_head.act(afford["logits"])
        return ret
