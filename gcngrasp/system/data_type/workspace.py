from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class IrregularWorkspace:
    pcd: np.ndarray  # hand
    r_decay: float = 1.
    is_border: bool = True

    def __post_init__(self):
        if not isinstance(self.pcd, np.ndarray):
            self.pcd = np.load(self.pcd)

    def rectify_lookat(self,
                       pos: np.ndarray,
                       focus: np.ndarray,
                       dist_min: float) -> np.ndarray:
        """ Rectify the position to fit the workspace constraints """
        assert pos.shape[-1] == 3

        f2c = pos - focus
        f2c_l = np.linalg.norm(f2c, axis=-1, keepdims=True)
        f2c_v = f2c / f2c_l

        f2p = self.sample_lookat(focus, dist_min) - focus
        f2p_l = np.linalg.norm(f2p, axis=-1, keepdims=True)
        f2p_v = f2p / f2p_l

        # select the most similar view
        i = np.argmax(f2c_v @ f2p_v.T, axis=-1)
        return focus + f2p_v[i] * np.minimum(f2c_l, f2p_l[i] * self.r_decay)

    def sample_lookat(self,
                      focus: np.ndarray,
                      dist_min: float,
                      n: int = None,
                      elev_range: tuple = None) -> np.ndarray:
        pcd = self.pcd - focus
        dist = np.linalg.norm(pcd, axis=-1)

        # distance
        if dist_min is not None:
            _mask = dist > dist_min / self.r_decay
            pcd = pcd[_mask]

        # elevation
        if elev_range is not None:
            elev_range = np.tan(elev_range)
            elev = pcd[:, 2] / np.linalg.norm(pcd[:, :2], axis=-1)
            pcd = pcd[(elev > elev_range[0]) & (elev < elev_range[1])]

        # unified radius
        if self.is_border:
            pcd = dist_min * pcd / np.linalg.norm(pcd, axis=-1, keepdims=True)

        # farthest point sampling
        if n is not None and n < pcd.shape[0]:
            from ...api.utils import o3d_extension as o3de
            pcd = pcd[o3de.farthest_point_sampling(pcd, n)]

        return pcd + focus

    def visualize(self, fig=None):
        """ Visualize the workspace """
        fig = fig or plt.subplot(projection="3d")
        fig.scatter(*self.pcd.T, s=1, c="b", alpha=0.5)
        plt.axis("equal")
        return fig


if __name__ == '__main__':
    import os

    os.chdir("../../..")
    ws = IrregularWorkspace(pcd="config/system/franka-ws.npy", r_decay=.9)

    # ra.workspace.visualize(), plt.show()
    pcd = ws.sample_lookat(np.array([.3, 0, 0]), dist_min=.2, elev_range=(0, np.pi / 3), n=20)
    fig = plt.subplot(projection="3d")
    fig.scatter(*pcd.T, s=1, c="r", alpha=0.5)
    plt.axis("equal")
    # ws.visualize(fig)
    plt.show()
