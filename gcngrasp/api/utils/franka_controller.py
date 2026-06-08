from __future__ import annotations

import numpy as np
import torch

import polymetis
import torchcontrol as toco
from . import pose_tf

MAX_OPENING = .0889
DEFAULT_SPEED = 200
DEFAULT_FORCE = 100

THRESH_POS = .005
THRESH_ROT = .05
THRESH_JOINT = .01


class PolymetisRobot(polymetis.RobotInterface):

    def joints(self) -> np.ndarray:
        return self.get_joint_positions().numpy()

    def move(self,
             goal: np.ndarray,
             vel: np.ndarray = None,
             success_always: bool = True) -> bool:
        if vel is None:

            # ee_pose
            if goal.ndim == 2:
                vec = torch.from_numpy(pose_tf.SE3.mat2vec(goal, scalar_first=False))
                self.move_to_ee_pose(vec[:3], vec[3:])
                return success_always or pose_tf.SE3.isin(
                    goal[None], self.Teb()[None], thresh_pos=THRESH_POS, thresh_rot=THRESH_ROT).item()

            # joint
            elif goal.ndim == 1:
                goal = torch.as_tensor(goal)
                self.move_to_joint_positions(goal)

        else:
            # traj
            pos_vel = list(map(torch.as_tensor, (goal, vel)))
            goal = pos_vel[0][-1]
            policy = toco.policies.JointTrajectoryExecutor(
                joint_pos_trajectory=pos_vel[0],
                joint_vel_trajectory=pos_vel[1],
                Kq=self.Kq_default,
                Kqd=self.Kqd_default,
                Kx=self.Kx_default,
                Kxd=self.Kxd_default,
                robot_model=self.robot_model,
                ignore_gravity=self.use_grav_comp,
            )
            self.send_torch_policy(policy)

        return success_always or np.abs(goal.numpy() - self.joints()).max() < THRESH_JOINT

    def Teb(self) -> np.ndarray:
        pos, quat = self.get_ee_pose()
        vec = torch.cat([pos, quat]).numpy()
        return pose_tf.SE3.vec2mat(vec, scalar_first=False)


class PolymetisGripper(polymetis.GripperInterface):

    def grip(self,
             is_close: bool,
             speed: float = DEFAULT_SPEED,
             force: float = DEFAULT_FORCE):
        kwargs = dict(speed=speed, force=force)
        self.goto(width=MAX_OPENING, **kwargs) if is_close else self.grasp(**kwargs)
        return True
