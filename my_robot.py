from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import zmq
from tqdm import tqdm

from gcngrasp.api import Path
from gcngrasp.api.utils import func_invoker
from gcngrasp.api.utils import pose_tf
from gcngrasp.api.utils.camera import Frame, Pinhole, ObservationSequence
from gcngrasp.api.utils.realsense import RGBDCamera
from gcngrasp.api.utils.zmq_socket import ZmqSocket

# interface
try:
    from gcngrasp.api.utils.franka_controller import (
        PolymetisRobot, PolymetisGripper, DEFAULT_SPEED, DEFAULT_FORCE, MAX_OPENING)

    RobotType = PolymetisRobot
    GripperType = PolymetisGripper

    from Robotiq2F85Driver.Robotiq2F85Driver import Robotiq2F85Driver


    class RobotiqGripper(Robotiq2F85Driver):

        def __init__(self, serial_number="DAAIT8BB"):
            super().__init__(serial_number)
            self.reset()

        def grip(self,
                 is_close: bool,
                 speed: float = DEFAULT_SPEED,
                 force: float = DEFAULT_FORCE):
            self.go_to(opening=0 if is_close else MAX_OPENING, speed=speed, force=force)
            return True


    GripperType = RobotiqGripper

except ImportError as e:
    print(e)


# robot
@dataclass
class RobotController:
    camera: Pinhole

    def __post_init__(self):
        self.gripper = GripperType()
        self.robot = RobotType()
        self.robot.go_home()
        self.rs = RGBDCamera(device_id=0)

    @property
    def functions(self) -> dict:
        return {
            "get_frame": self.get_frame,
            "org_frame": self.org_frame,

            "Teb": self.robot.Teb,
            "joints": self.robot.joints,
            "go_home": lambda: (self.gripper.reset(), self.robot.go_home()),
            "move": self.robot.move,

            "grip": self.gripper.grip,
            "grip_reset": lambda: self.gripper.reset() or True,
        }

    def get_frame(self) -> Frame:
        (t, color, depth), Teb = self.rs(), self.robot.Teb()
        assert Teb is not None
        pose_tf.TFmanager.register(Teb, "hand", "base")
        return Frame(
            camera=self.camera,
            color=color,
            depth=depth,
            Tcw=pose_tf.TFmanager["camera", "world"],
        )

    def org_frame(self):
        frame, Teb = self.rs(), self.robot.Teb()
        return frame, dict(transform={"T": Teb, "src": "hand", "dst": "base"})


@dataclass
class CuRoboWrapper:
    invoker: func_invoker.BaseInvoker
    z_ground: float = .05
    vel_scale: float = .1

    def __post_init__(self):
        from gcngrasp.api.utils.curobo_planning import CuRoboPlanner, CuRoboController

        self.planner = CuRoboPlanner(interp_dt=8e-4, scene_cfg=self.z_ground, max_goalset=100)
        self.controller = CuRoboController(
            self.planner,
            func_joints=self.invoker.joints,
            func_grip=self.invoker.grip,
            func_move=lambda js: self.invoker.move(js.position.cpu(), js.velocity.cpu() * self.vel_scale) if js else False
        )

    def __getattr__(self, item):
        try:
            return getattr(self.controller, item)
        except AttributeError:
            return getattr(self.invoker, item)


# client
def get_robot_client(addr: str = None,
                     camera: Pinhole = None,
                     ) -> func_invoker.BaseInvoker:
    return func_invoker.ZmqSocketCLient(
        sock=ZmqSocket(zmq.REQ, addr=addr, queue_size=10)
    ) if addr else func_invoker.LocalInvoker(functions=RobotController(camera).functions)


# server
def get_robot_server(addr: str,
                     camera: Pinhole,
                     ) -> func_invoker.ZmqSocketServer:
    assert camera is not None
    return func_invoker.ZmqSocketServer(
        functions=RobotController(camera).functions,
        sock=ZmqSocket(zmq.REP, addr=addr, queue_size=10),
        loginfo=print,
    )


if __name__ == '__main__':
    pose_tf.TFmanager.load_yaml("config/system/franka_tf.yaml")
    camera = Pinhole.from_yaml("config/system/RS-D405.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", type=str, default="", help="server address")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--ground", type=float, default=.05)
    args = parser.parse_args()

    # client
    if args.addr:
        robot_ctrl = get_robot_client(addr=f"{args.addr}:{args.port}")
        robot_ctrl = CuRoboWrapper(robot_ctrl, z_ground=args.ground)

        # --- collect data ---
        obs_seq = ObservationSequence(Path("runs/scene01"))
        file = Path("runs/Tebs.npy")
        if file.exists():
            Tebs = file.numpy()
            for i, T in enumerate(tqdm(Tebs[:])):
                robot_ctrl.go_home(), robot_ctrl.move(T)
                input("wait")
                args, kwargs = robot_ctrl.org_frame()
                Path(obs_seq.root / f"{i}.bin").binary(kwargs)

            for i, T in enumerate(tqdm(Tebs[:])):
                robot_ctrl.go_home(), robot_ctrl.move(T)
                time.sleep(.5)
                args, kwargs = robot_ctrl.org_frame()
                obs_seq.save_frame(*args, **kwargs)

    # server
    else:
        robot_ctrl = get_robot_server(addr=f"0.0.0.0:{args.port}", camera=camera)
        robot_ctrl.main(), exit()
