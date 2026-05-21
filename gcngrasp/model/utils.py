import pickle

import numpy as np
import torch


def load_pickle(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def visualize_grasp(grasp: torch.Tensor,
                    pcd: torch.Tensor,
                    n_samples: int = 5,
                    group_pcd: np.ndarray = None,
                    connections: np.ndarray = None):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    n_gcp = grasp.shape[-2]
    connections = np.stack([np.arange(n_gcp - 1), np.arange(1, n_gcp)], axis=-1) if connections is None else connections

    # random sample
    i = np.random.randint(0, len(grasp), n_samples)
    grasp, pcd = grasp.cpu().numpy()[i], pcd.cpu().numpy()[i]
    if group_pcd is not None: group_pcd = group_pcd.cpu().numpy()[i]

    for i, (g, p) in enumerate(zip(grasp, pcd)):
        fig = plt.subplot(projection="3d")

        # pcd
        c_i = np.full(p.shape[0], -1, dtype=int)
        if group_pcd is not None:
            for j, idx in enumerate(group_pcd[i]):
                c_i[idx] = j
        mask = c_i >= 0
        fig.scatter(*p[~mask].T, c="gray", alpha=.1)
        fig.scatter(*p[mask].T, c=cmap(c_i[mask]))

        # gripper
        for line in g[connections]:
            fig.plot(*line.T, c="orange")

        plt.axis("equal"), plt.show()


def normalize_pcd(pcd: torch.Tensor,
                  grasp: torch.Tensor,
                  scaling: bool = True
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    normalizer = PcdNormalizer(pcd, scaling=scaling)
    return tuple(map(normalizer.forward, [pcd, grasp]))


class PcdNormalizer:

    def __init__(self,
                 pcd: torch.Tensor,
                 scaling: bool = True):
        """
        :param pcd: point cloud [B, N, 3]
        :param scaling: whether to scale the point cloud
        """
        assert pcd.ndim == 3
        self.mean = pcd.mean(dim=1)
        self.scale = torch.norm(pcd - self.mean[:, None], dim=-1).max(dim=-1).values \
            if scaling else torch.ones(len(pcd), dtype=pcd.dtype, device=pcd.device)

    def forward(self,
                x: torch.Tensor):
        shape = [len(x)] + [1] * (x.ndim - 2) + [-1]
        return (x - self.mean.view(shape)) / self.scale.view(shape)

    def backward(self,
                 x: torch.Tensor):
        shape = [len(x)] + [1] * (x.ndim - 2) + [-1]
        return x * self.scale.view(shape) + self.mean.view(shape)
