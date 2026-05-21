from dataclasses import dataclass

from . import planner
from .data_type.action import Action
from .scene import Scene
from ..api.utils import Timer


@dataclass
class System:
    config: dict
    cfg_nbv_proxy: dict = None
    timer_momentum: float = .5

    def __post_init__(self):
        if self.cfg_nbv_proxy:
            from .nbv_server import NBVProxy
            self.planner = NBVProxy(self.cfg_nbv_proxy)
        else:
            self.planner = planner.BasePlanner.from_config(self.config["planner"])
        Timer.set_momentum(self.timer_momentum)

    def get_action(self,
                   scene: Scene,
                   plan_nbv: bool = True) -> Action:
        """ Plan next best view or search for grasp """
        tog, _ = scene.tog_poses
        act = Action.from_grasp(tog[tog.mask_by_relative_thresh(.9)].Tgx)

        # next best view
        if plan_nbv:
            nbv = self.planner.solve(scene)
            if nbv: act = nbv[None]

        Timer.to_dict(log=True)
        return act
