from __future__ import annotations

import open3d as o3d

from gcngrasp import Path
from gcngrasp.api.utils.camera import Frame
from gcngrasp.system import process_config
from gcngrasp.system import virtual_robot
from gcngrasp.system.api import get_da3_mapper
from gcngrasp.system.data_type.workspace import IrregularWorkspace
from gcngrasp.system.system import System, Scene

ASSETS = list(Path("assets").iterdir())
WORKSPACE = IrregularWorkspace("config/system/franka_ws.npy", r_decay=.9)


def offline_test(config: dict,
                 frame: Frame,
                 args: list):
    # config
    frame.Tcw[:3, 3] += [0.4, 0, 0.1]
    frames = [frame]

    # initial
    mapper = get_da3_mapper(enable_main=len(frames) > 1)
    scene = Scene(config, target_obj=args[0], tog_model_args=args[1:3], workspace=WORKSPACE)
    system = System(config)

    # update
    for i, f in enumerate(frames):
        scene.integrate(f, mapper=mapper)
        act = system.get_action(scene, plan_nbv=True)[0]
        print(act)
        o3d.visualization.draw_geometries(scene.to_o3d(next_Tcw=act.Tcw, coord=True))


if __name__ == '__main__':
    CONFIG = process_config("config/system/params.yaml")

    offline_test(CONFIG, *virtual_robot.frame_from_folder(ASSETS[1]))
