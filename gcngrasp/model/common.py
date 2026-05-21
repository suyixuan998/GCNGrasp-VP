from functools import lru_cache
from typing import List, Optional

import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch import nn

from . import utils

sum_ = lambda x: sum(x[1:], x[0])
make_divisible = lambda x, divisor=4: np.maximum(np.round(x / divisor).astype(np.int64), 1) * divisor


def apply_width_multiplier(channels: list,
                           w: float,
                           divisor: int = 4) -> list:
    """ Apply width multiplier to channels. """
    return [
        apply_width_multiplier(c, w, divisor) if isinstance(c, list) else make_divisible(c * w, divisor=divisor)
        for c in channels
    ]


def fuse_modules(model: nn.Module):
    for m in filter(lambda m: isinstance(m, LinearBnAct) and not m.deploy, model.modules()):
        torch.ao.quantization.fuse_modules(m, ["linear", "bn"], inplace=True)


class LinearBnAct(nn.Module):
    """ Linear - BN - Act """
    deploy = property(lambda self: isinstance(self.bn, nn.Identity))
    LinearType = nn.Linear
    BnType = nn.BatchNorm1d

    def __init__(self, c1, c2, act: Optional[nn.Module] = nn.ReLU, **linear_kwargs):
        super().__init__()
        self.c2 = c2
        self.linear_kwargs = linear_kwargs
        self.linear = self.LinearType(c1, c2, bias=False, **linear_kwargs)
        self.bn = self.BnType(c2)
        self.act = act() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.linear(x)))

    @classmethod
    def create_mlp(cls, c1, c2s, linear_output=False, **kwargs):
        """ Create MLP. """
        layers = nn.Sequential()
        for c2 in (c2s[:-1] if linear_output else c2s):
            layers.append(cls(c1, c2, **kwargs))
            c1 = c2
        if linear_output:
            layers.append(cls.LinearType(c1, c2s[-1], bias=True, **layers[-1].linear_kwargs))
        layers.c2 = c2s[-1]
        return layers


class _ConvBnActNd(LinearBnAct):
    """ Conv - BN - Act """

    @staticmethod
    def auto_pad(k, s=1, d=1):
        # (k - 1) // 2 * d: 1st-center -> [0, 0]
        # (s - 1) // 2: 1st-center -> [s/2, s/2]
        return max(0, (k - 1) // 2 * d - (s - 1) // 2)

    def __init__(self, c1, c2, k=3, s=1, g=1, d=1,
                 act: Optional[nn.Module] = nn.ReLU, ctrpad=True):
        assert k & 1, "The convolution kernel size must be odd"
        padding = self.auto_pad(k, s if ctrpad else 1, d)
        super().__init__(c1, c2, act=act,
                         kernel_size=k, stride=s, padding=padding, groups=g, dilation=d)


class ConvBnAct1d(_ConvBnActNd):
    LinearType = nn.Conv1d
    BnType = nn.BatchNorm1d


class ConvBnAct2d(_ConvBnActNd):
    LinearType = nn.Conv2d
    BnType = nn.BatchNorm2d


class GCNConvBnAct(LinearBnAct):
    LinearType = torch_geometric.nn.GCNConv
    BnType = torch_geometric.nn.BatchNorm

    def __init__(self, c1, c2, act: Optional[nn.Module] = nn.ReLU):
        super().__init__(c1, c2, act=act, improved=False)

    def forward(self, graphs: torch_geometric.data.Batch):
        graphs.x = self.linear(graphs.x, graphs.edge_index)
        graphs.x = self.act(self.bn(graphs.x))
        return graphs


class FocalWeight(nn.Module):

    def __init__(self,
                 alpha: float = .5,
                 gamma: float = 2.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.scale = 2 / .5 ** self.gamma

    def forward(self, y_score, y_true):
        with torch.no_grad():
            alpha_t = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
            gamma_t = ((1 - y_score) * y_true + y_score * (1 - y_true)) ** self.gamma
            return self.scale * alpha_t * gamma_t


class KnowledgeGraph(nn.Module):

    def __init__(self,
                 c1: int,
                 config: dict):
        super().__init__()
        graph, _seeds = utils.load_pickle(config["graph_data"])
        self.map_obj2cls: dict = utils.load_pickle(config["map_obj2cls"])[-1]

        self.graph: nx.DiGraph = nx.convert_node_labels_to_integers(graph, label_attribute='name')
        objcls_list: List[str] = utils.load_pickle(config["class_list"])

        # look up node
        self.map_cls2node: dict = {
            cls: [i for i, node in self.graph.nodes.data() if node["name"] == cls].pop() for cls in objcls_list
        }
        self.map_task2node: dict = {
            node["name"]: i for i, node in self.graph.nodes.data() if node["type"] == "task"
        }

        # embedding
        self.embedding = nn.Embedding(len(self), c1)
        self.c2 = c1 + 1

    def __len__(self):
        return len(self.graph)

    @lru_cache(maxsize=None)
    def ego_graph(self,
                  node_id: int,
                  hop: int = 3):
        return nx.ego_graph(self.graph, node_id, radius=hop, undirected=True)

    def example_input(self, B: int):
        """ Generate example input """
        import random

        obj_embedding = torch.rand([B, self.embedding.embedding_dim])
        objs = random.choices(list(self.map_cls2node), k=B)
        tasks = random.choices(list(self.map_task2node), k=B)
        return obj_embedding, objs, tasks

    def name2id(self,
                obj: str,
                task: str):
        # print(set(self.map_obj2cls) | set(self.map_cls2node))
        try:
            obj = self.map_obj2cls.get(obj, obj)
            obj_id = self.map_cls2node[obj]
        except KeyError:
            raise KeyError(f"Available object: {list(self.map_cls2node)}")

        try:
            task_id = self.map_task2node[task]
        except KeyError:
            raise KeyError(f"Available task: {list(self.map_task2node)}")
        return obj_id, task_id

    def process_input(self,
                      obj_embedding: torch.Tensor,
                      obj: str,
                      task: str) -> torch_geometric.data.Data:
        obj_id, task_id = self.name2id(obj, task)
        graph = nx.compose(*map(self.ego_graph, [obj_id, task_id]))

        x = torch.cat([self.embedding.weight, obj_embedding[None]], dim=0)
        flag = torch.zeros([len(self) + 1], device=x.device)
        flag[[obj_id, task_id, len(self)]] = 1
        x = torch.cat([x, flag[:, None]], dim=1)

        edges = list(graph.edges) + [(obj_id, len(self))]
        edges = torch.tensor(edges, device=x.device, dtype=torch.long).T
        return torch_geometric.data.Data(x=x, edge_index=torch_geometric.utils.to_undirected(edges))


if __name__ == '__main__':
    B, N, g = 4, 10, 8

    x = torch.rand([B, 16, N])
    y = torch.randint(-1, g, [B, N])
