from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

import curobo.scene as cusc
from curobo._src.state.state_joint_ops import stack_joint_states
from curobo.config_io import resolve_config, join_path, load_yaml
from curobo.content import get_scene_configs_path
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsResult
from curobo.kinematics import Kinematics
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg, GraspPlanResult
from curobo.trajectory_optimizer import TrajectoryOptimizerResult
from curobo.types import DeviceCfg, Pose, JointState, GoalToolPose

os.environ["CUROBO_TORCH_COMPILE_DISABLE"] = "1"

# cuda settings
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


def create_table(h: float,
                 l: float = 1.5,
                 b: float = .3,
                 index: tuple[int] = (0,)) -> cusc.Scene:
    assert h > 0
    a = (l - b) / 2
    c = (a + b) / 2

    q = [1, 0, 0, 0]
    cuboid = [
        cusc.Cuboid(name="tb_front", pose=[c, 0, h / 2] + q, dims=[a, l, h]),
        cusc.Cuboid(name="tb_back", pose=[-c, 0, h / 2] + q, dims=[a, l, h]),
        cusc.Cuboid(name="tb_left", pose=[0, c, h / 2] + q, dims=[b, a, h]),
        cusc.Cuboid(name="tb_right", pose=[0, -c, h / 2] + q, dims=[b, a, h]),
    ]
    return cusc.Scene(cuboid=[cuboid[i] for i in index])


def any_to_curobo(cutype, data, *args, **kwargs):
    if cutype is type(data): return data

    # Any -> Pose
    if cutype is Pose:
        func = {
            7: Pose.from_batch_list,
            4: Pose.from_matrix,
        }[data.shape[-1]]
        return func(data, **kwargs)

    else:
        raise TypeError(f"Unsupported curobo type: {cutype}.")


def curobo_to_ros(cudata, *args, **kwargs):
    # curobo.types.state.JointState -> trajectory_msgs.msg.JointTrajectory
    if isinstance(cudata, JointState):
        import rospy
        import trajectory_msgs.msg

        dt = kwargs["dt"]
        msg = trajectory_msgs.msg.JointTrajectory()
        msg.joint_names = cudata.joint_names
        msg.header.frame_id = msg.joint_names[0].split("_")[0] + "_link0"

        msg.points = [
            trajectory_msgs.msg.JointTrajectoryPoint(positions=p, velocities=v, accelerations=a, time_from_start=t)
            for p, v, a, t in
            zip(cudata.position.tolist(), cudata.velocity.tolist(), cudata.acceleration.tolist(), map(rospy.Duration, np.arange(dt, len(cudata), dt)))
        ]
        return msg

    else:
        raise TypeError(f"Unsupported curobo type: {type(cudata)}.")


def process_result(result, *args, **kwargs):
    success = result.success
    if not success.any(): return

    # InverseKinematicsResult -> JointState
    if isinstance(result, InverseKinematicsResult):
        return result.js_solution[success]

    # TrajectoryOptimizerResult -> JointState
    elif isinstance(result, TrajectoryOptimizerResult):
        plan = result.get_interpolated_plan()[0, 0]
        planner, vis_cfg = args
        if vis_cfg: vis_cfg.visualize(plan.position, planner).show()
        return plan

    # GraspPlanResult -> list[JointState]
    elif isinstance(result, GraspPlanResult):
        plans = [stack_joint_states(result.approach_interpolated_trajectory, result.grasp_interpolated_trajectory)[0, 0]]
        if result.lift_trajectory:
            plans.append(result.lift_interpolated_trajectory[0, 0])

        planner, vis_cfg = args
        if vis_cfg:
            traj = stack_joint_states(result.approach_trajectory, result.grasp_trajectory)
            if result.lift_trajectory:
                traj = stack_joint_states(traj, result.lift_trajectory)
            vis_cfg.visualize(traj[0, 0].position, planner).show()
        return plans

    else:
        raise TypeError(f"Unsupported curobo type: {type(result)}.")


@dataclass
class PlanVisConfig:
    n_vis: int = 15
    cmap: str = "autumn_r"
    alpha_start: float = .5
    alpha_traj: float = .1
    alpha_end: float = 1.

    def __post_init__(self):
        assert self.n_vis >= 2

    def sample_indices(self, n: int) -> tuple[torch.Tensor, np.ndarray]:
        """ Sample indices for visualization. """
        i = torch.linspace(0, n - 1, self.n_vis).long()
        colors = plt.get_cmap(self.cmap)(i / i[-1])
        colors[0, -1], colors[1:-1, -1], colors[-1, -1] = self.alpha_start, self.alpha_traj, self.alpha_end
        return i, colors

    def visualize(self,
                  positions: torch.Tensor,
                  planner: "CuRoboPlanner") -> trimesh.Scene:
        i, colors = self.sample_indices(len(positions))

        # generate scene
        scene: list[cusc.Scene] = [planner.scene_robot(q[None], as_trimesh=False) for q in positions[i]]
        for s, c in zip(scene, colors): s.add_color(c)
        scene.insert(0, scene.pop())
        scene.append(planner.scene_world(as_trimesh=False))

        # to trimesh
        return trimesh.util.concatenate([cusc.Scene.get_scene_graph(s) for s in scene])


@dataclass
class CuRoboPlanner:
    """ reference: https://curobo.org/ """
    robot_cfg: str = "franka.yml"
    scene_cfg: cusc.Scene = .05
    device_cfg: DeviceCfg = field(default_factory=DeviceCfg)

    # Motion planner parameters
    interp_dt: float = 5e-4
    max_goalset: int = 1

    def __post_init__(self):
        if isinstance(self.scene_cfg, (int, float)):
            self.scene_cfg = create_table(h=self.scene_cfg)
        elif isinstance(self.scene_cfg, str):
            self.scene_cfg = load_yaml(resolve_config(join_path(get_scene_configs_path(), self.scene_cfg)))
        if not isinstance(self.scene_cfg, cusc.Scene):
            self.scene_cfg = cusc.Scene.create(self.scene_cfg)
        self.scene_cfg = cusc.Scene.create_collision_support_world(self.scene_cfg)

        assert len(self.motion_planner.tool_frames) == 1, "Only single tool frame is supported."

    @property
    def default_js(self) -> torch.Tensor:
        return self.motion_planner.default_joint_state.position

    @classmethod
    def generate_ws_for_lookat(
            cls,
            dst_file: str,
            focus: np.ndarray,
            Tec: np.ndarray,
            azim_limit: tuple[float, float] = (-np.pi, np.pi),
            elev_limit: tuple[float, float] = (0, np.pi / 2 - 0.01),
            tol_pos: float = 0.01,
            tol_rad: float = 0.06,
            radius_min: float = 0.05,
            cmap: str = "rainbow",
    ):
        """
        Estimates the workspace point cloud of the robot arm's end effector and saves it to a file.
        NOTE: Execute `roslaunch panda_moveit_config demo.launch robot:=panda` to start the MoveIt node.

        :param dst_file: path to save the point cloud (.npy file)
        :param focus: focus point
        :param azim_limit: azimuth angle range (rad)
        :param elev_limit: elevation angle range (rad)
        :param tol_pos: position increment/tolerance
        :param tol_rad: angle increment/tolerance
        :param radius_min: minimum distance from the focus
        """
        import logging
        import itertools
        from tqdm import trange
        from . import pose_tf

        logging.basicConfig(level=logging.FATAL, format="[%(levelname)s] %(message)s")
        cmap = plt.get_cmap(cmap)

        planner = cls(scene_cfg={"cuboid": {
            "focus": {"dims": [radius_min / 2] * 3, "pose": np.concatenate([focus, [1, 0, 0, 0]])}
        }})

        # generate sampled azimuth and elevation angles
        azim = np.linspace(*azim_limit, round(np.diff(azim_limit).item() / tol_rad))
        elev = np.linspace(*elev_limit, round(np.diff(elev_limit).item() / tol_rad))
        azim, elev = map(np.ravel, np.meshgrid(azim, elev))

        # shuffle the angles
        n = len(azim)
        i = np.random.permutation(n)
        azim, elev = azim[i], elev[i]

        # unit direction vectors for each sample
        lookat = np.stack([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), - np.sin(elev)], axis=-1).astype(np.float32)
        radius = np.zeros(n, dtype=np.float32)

        # incrementally increase radius and check reachability
        fig = plt.subplot(projection="3d")
        tbar = trange(n, desc="Estimating workspace")
        for i in tbar:
            for r in itertools.count(radius_min, tol_pos):
                pos = focus - lookat[i] * r
                Tcb = pose_tf.lookat_to_Tcw(pos, focus)
                res = planner.plan(planner.default_js[None], goal=Pose.from_matrix(Tcb @ Tec))

                # record the last reachable radius
                if not res: break
                radius[i] = r

            # update the progress bar
            valid_rate = (radius[:i + 1] > 0).mean()
            tbar.set_postfix({"valid_rate": round(valid_rate, 4)})

            # compute all reachable spatial coordinates
            pcd = focus - lookat[:i + 1] * radius[:i + 1, None]
            pcd = pcd[radius[:i + 1] > 0]
            np.save(dst_file, pcd)

            # visualize the current solution
            plt.cla()
            fig.scatter(*pcd.T, s=20, c=cmap(pcd[:, 2]))
            fig.view_init(elev=60, azim=-120)
            plt.axis("equal")
            plt.pause(1e-3)

    @cached_property
    def ik_solver(self) -> InverseKinematics:
        return self.motion_planner.ik_solver

    @cached_property
    def kinematics(self) -> Kinematics:
        return self.motion_planner.kinematics

    @cached_property
    def motion_planner(self) -> MotionPlanner:
        mp_cfg = MotionPlannerCfg.create(
            robot=self.robot_cfg,
            scene_model=self.scene_cfg,
            device_cfg=self.device_cfg,
            max_goalset=self.max_goalset,
        )
        mp_cfg.trajopt_solver_config.interpolation_dt = self.interp_dt
        mp = MotionPlanner(mp_cfg)
        mp.warmup(num_warmup_iterations=5)
        return mp

    def plan(self,
             q_start: torch.Tensor,
             goal: torch.Tensor,
             vis_cfg: PlanVisConfig = None) -> JointState:
        q_start = JointState.from_position(q_start[None]).to(self.device_cfg)
        goal = self.process_goal(goal[None])

        # function
        func = {
            JointState: self.motion_planner.plan_cspace,
            GoalToolPose: self.motion_planner.plan_pose,
        }[type(goal)]

        # process
        result = func(goal, q_start)
        return process_result(result, self, vis_cfg)

    def plan_goalset(self,
                     q_start: torch.Tensor,
                     goal: torch.Tensor,
                     vis_cfg: PlanVisConfig = None) -> JointState:
        q_start = JointState.from_position(q_start[None]).to(self.device_cfg)
        goal = self.process_goal(goal)

        # process
        result = self.motion_planner.plan_pose(q_start, goal)
        return process_result(result, self, vis_cfg)

    def plan_grasp(self,
                   q_start: torch.Tensor,
                   goal: torch.Tensor,
                   d_approach: float = .1,
                   d_retract: float = .1,
                   is_lift: bool = False,
                   vis_cfg: PlanVisConfig = None):
        q_start = JointState.from_position(q_start[None]).to(self.device_cfg)
        goal = self.process_goal(goal)

        # process
        result = self.motion_planner.plan_grasp(
            goal, q_start,
            grasp_approach_offset=-d_approach,
            grasp_lift_offset=d_retract * (1 if is_lift else -1),
            plan_grasp_to_lift=d_retract > 0,
            grasp_lift_in_tool_frame=not is_lift,
        )
        return process_result(result, self, vis_cfg)

    def process_goal(self,
                     goal: torch.Tensor) -> GoalToolPose:
        dim = goal.shape[-1]
        if dim == 4:
            goal = any_to_curobo(Pose, goal).to(self.device_cfg)
            return GoalToolPose.from_poses(
                {k: v for k, v in zip(self.tool_frames, [goal])},
                num_goalset=len(goal)
            )
        elif dim == 7:
            return JointState.from_position(goal).to(self.device_cfg)
        else:
            raise ValueError(f"Invalid goal shape: {goal.shape}")

    def sample_states(self,
                      n: int) -> torch.Tensor:
        return self.ik_solver.sample_configs(n)

    def scene_robot(self,
                    q: torch.Tensor,
                    as_trimesh: bool = True) -> cusc.Scene | trimesh.Scene:
        q = self.device_cfg.to_device(q)[..., :self.kinematics.dof]
        scene = cusc.Scene(sphere=self.kinematics.get_robot_as_spheres(q)[0])
        return cusc.Scene.get_scene_graph(scene) if as_trimesh else scene

    def scene_world(self,
                    as_trimesh: bool = True) -> cusc.Scene | trimesh.Scene:
        return cusc.Scene.get_scene_graph(self.scene_cfg) if as_trimesh else self.scene_cfg

    def solve_fk(self,
                 q: torch.Tensor) -> torch.Tensor:
        """ Forward kinematics """
        q = JointState.from_position(q)
        kin_state = self.kinematics.compute_kinematics(q).tool_poses
        return kin_state.get_link_pose(self.tool_frames[0]).get_matrix()

    def solve_ik(self,
                 goal: Pose) -> InverseKinematicsResult:
        """ Inverse kinematics """
        goal = self.process_goal(goal)
        return self.ik_solver.solve_pose(goal)

    @property
    def tool_frames(self) -> list[str]:
        return self.motion_planner.tool_frames


@dataclass
class CuRoboController:
    planner: CuRoboPlanner
    func_joints: Callable[[], torch.Tensor]
    func_grip: Callable[[bool], bool]
    func_move: Callable[[JointState], bool]

    def go_home(self) -> bool:
        return self.move(self.planner.default_js)

    def grasp(self, Tebs: torch.Tensor, **kwargs) -> bool:
        Tebs = torch.as_tensor(Tebs)
        plans = self.planner.plan_grasp(self.joints(), Tebs, **kwargs)
        if not plans: return False
        return self.func_move(plans[0]) and self.func_grip(True) and (
                len(plans) == 1 or self.func_move(plans[1])
        )

    def joints(self) -> torch.Tensor:
        return torch.as_tensor(self.func_joints())

    def move(self, Teb_or_joints: torch.Tensor, **kwargs) -> bool:
        Teb_or_joints = torch.as_tensor(Teb_or_joints)
        plan = self.planner.plan(self.joints(), Teb_or_joints, **kwargs)
        if plan is None: return False
        return self.func_move(plan)

    def place(self, Tebs: torch.Tensor, **kwargs) -> bool:
        Tebs = torch.as_tensor(Tebs)
        plans = self.planner.plan_grasp(self.joints(), Tebs, is_lift=False, **kwargs)
        if not plans: return False
        return self.func_move(plans[0]) and self.func_grip(False) and (
                len(plans) == 1 or self.func_move(plans[1])
        )

    def Teb(self) -> torch.Tensor:
        return torch.inverse(self.planner.solve_fk(self.joints())[0])


if __name__ == '__main__':
    print_and_ret = lambda x: print("success!") or True

    planner = CuRoboPlanner(scene_cfg=.05, interp_dt=5e-4, max_goalset=10)
    vis_cfg = PlanVisConfig()
    controller = CuRoboController(planner, func_joints=lambda: planner.default_js,
                                  func_move=print_and_ret, func_grip=print_and_ret)
    # vis_cfg.visualize(planner.default_js[None], planner).show()

    for i in range(1):
        q = planner.sample_states(planner.max_goalset)
        goal = planner.solve_fk(q)

        controller.move(goal[2], vis_cfg=vis_cfg)
        # planner.plan(q[0], q[1], vis_cfg=vis_cfg)
        controller.grasp(goal, vis_cfg=vis_cfg, is_lift=True, d_retract=.1)
