from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import sklearn


@dataclass
class BaseIGField:
    pcd: np.ndarray
    ig: np.ndarray

    center: np.ndarray = None
    radius: float = None
    normal: np.ndarray = None

    field_prev: "BaseIGField" = None
    idx_prev: np.ndarray = None

    def __post_init__(self):
        if self.center is None:
            self.center = (self.pcd.min(axis=0) + self.pcd.max(axis=0)) / 2
        if self.radius is None:
            self.radius = np.linalg.norm(self.pcd_0m, axis=-1).max()
        if self.normal is None:
            self.normal = self.pcd_0m / (np.linalg.norm(self.pcd_0m, axis=-1, keepdims=True) + 1e-6)

    def __getitem__(self, item):
        return self.__class__(
            self.pcd[item], self.ig[item],
            center=self.center, radius=self.radius, normal=self.normal[item],
            field_prev=self, idx_prev=item
        )

    def __len__(self):
        return len(self.pcd)

    def clusters(self,
                 thresh_ig: float,
                 eps: float = .01,
                 visualize: bool = False) -> list["BaseIGField"]:
        thresh_ig = np.percentile(self.ig, thresh_ig * 100)
        mask = self.ig >= thresh_ig
        new = self[mask]

        cs = sklearn.cluster.DBSCAN(eps=eps, min_samples=3)
        cs.fit(new.pcd_0m)
        n = cs.labels_.max() + 1

        ret = [new[cs.labels_ == i] for i in range(n)]
        ret = sorted(ret, key=lambda f: f.ig.sum(), reverse=True)

        if visualize:
            cmap = plt.get_cmap("rainbow_r")
            fig = plt.subplot(projection="3d")
            fig.scatter(*self.pcd[~mask].T, c="gray", alpha=.1)
            for i, f in enumerate(ret):
                fig.scatter(*f.pcd.T, color=cmap(i / n))
            plt.axis("equal")
            plt.show()

        return ret

    def interp(self,
               pcd: np.ndarray,
               k: int = 4) -> "BaseIGField":
        knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors=k, metric="euclidean")
        knn.fit(self.pcd, self.ig)
        return self.__class__(pcd, knn.predict(pcd))

    @cached_property
    def pcd_0m(self) -> np.ndarray:
        return self.pcd - self.center

    def visualize(self, fig=None, alpha=1., cmap="rainbow"):
        fig = fig or plt.subplot(projection="3d")
        ig = (self.ig - self.ig.min()) / (self.ig.max() - self.ig.min())

        color = plt.get_cmap(cmap)(ig)
        color[..., -1] = alpha

        if self.ig.ndim == 1:
            fig.scatter(*self.pcd.T, c=color, s=1)
        elif self.ig.ndim == 2:
            fig.plot_surface(*self.pcd.transpose(2, 0, 1), facecolors=color)

        plt.axis("equal")
        return fig

    def to_o3d(self, cmap="rainbow") -> o3d.geometry.PointCloud:
        ig = (self.ig - self.ig.min()) / (self.ig.max() - self.ig.min())
        color = plt.get_cmap(cmap)(ig)[:, :3]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.pcd))
        pcd.colors = o3d.utility.Vector3dVector(color)
        return pcd
