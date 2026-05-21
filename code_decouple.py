from __future__ import annotations

import shutil
from pathlib import Path

from gcngrasp.api.utils.package_installer import Package

SOURCE = Path("gcngrasp")
PACKAGE = Package(
    SOURCE,
    dependency={
        # api
        "api": ["api/utils", "api/utils/func_invoker", "api/utils/path_extension"],
        "api/utils/franka_controller": ["api/utils/pose_tf"],
        "api/utils/grasp_pose": ["api/utils/o3d_extension"],

        # system
        "system/nbv_server": ["api/utils", "api/utils/camera", "api/utils/zmq_socket"],
    },
    packages=["api", "api/utils", "dataset", "model", "system", "system/data_type", "system/utils"]
)

DEP_MY_ROBOT = [
    "api", "api/utils/camera",
    "api/utils/pose_tf", "api/utils/realsense", "api/utils/zmq_socket"
]


def install_my_robot(root: Path):
    try:
        shutil.rmtree(root / "config")
    except FileNotFoundError:
        pass
    shutil.copytree("config/system", root / "config/system")
    shutil.copy("my_robot.py", root)


if __name__ == '__main__':
    # nbv_server
    ROOT = Path("3rd_party/src")
    for dst in ("active_ngf", "GauSS-MI/scripts"):
        dst = ROOT / dst
        assert dst.is_dir() and dst.exists()

        dst = PACKAGE.install(dst=dst, require=["api", "api/utils/grasp_pose", "system/nbv_server"])
        (dst / "__init__.py").write_text("")

    # robot_server
    pkg = ROOT / "robot_server"
    PACKAGE.install(dst=pkg, require=DEP_MY_ROBOT + ["api/utils/franka_controller"])
    install_my_robot(pkg)

    # robot_client
    pkg = ROOT / "robot_client"
    PACKAGE.install(dst=pkg, require=DEP_MY_ROBOT + ["api/utils/curobo_planning"])
    install_my_robot(pkg)
    shutil.copy("real_nbv.ipynb", pkg)
