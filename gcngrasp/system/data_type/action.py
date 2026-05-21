from dataclasses import dataclass

import numpy as np

from ...api.utils import pose_tf
from ...api.utils.grasp_pose import lift, pregrasp


def Tcw2Teb(Tcw: np.ndarray):
    Tec = pose_tf.TFmanager["hand", "camera"]
    Twb = pose_tf.TFmanager["world", "base"]
    return Twb @ Tcw @ Tec


@dataclass
class Action:
    Tcw: np.ndarray
    grasp: bool

    def __getitem__(self, item):
        return __class__(self.Tcw[item], grasp=self.grasp)

    @classmethod
    def from_grasp(cls,
                   Tgw: np.ndarray):
        Tcg = pose_tf.TFmanager["camera", "grasp"]
        # fix yaw
        Tgw[:, :3, :2] *= - (((Tgw @ Tcg)[:, 2, 1] > 0) * 2 - 1).reshape(-1, 1, 1)
        return cls(Tgw @ Tcg, grasp=True)

    @classmethod
    def from_lookat(cls,
                    pos: np.ndarray,
                    focus: np.ndarray):
        Tcw = pose_tf.lookat_to_Tcw(pos, focus)
        return cls(Tcw, grasp=False)

    @property
    def Teb(self) -> np.ndarray:
        return Tcw2Teb(self.Tcw)

    def to_Tebs(self,
                d_pre: float,
                d_lift: float) -> np.ndarray:
        Teb = self.Teb
        return np.stack([
            pregrasp(Teb, d=d_pre), Teb, lift(Teb, d=d_lift)
        ]) if self.grasp else Teb[None]
