from __future__ import annotations

import argparse

import colorama
import pandas as pd
import sklearn.metrics as skm
import trimesh
from logdir import LogDir

from gcngrasp.api import Path
from gcngrasp.system import process_config
from gcngrasp.system.scene import Scene
from gcngrasp.system.system import System
from gcngrasp.system.virtual_robot import VirtualRobot


def run(system: System,
        logdir: LogDir,
        data_path: Path,
        max_iter: int,
        seeds: list[int],
        init_frames: list[int] = None):
    # prepare
    logdir.copy(CONFIG, CONFIG.name)

    for robot in VirtualRobot.load_all_instances(data_path, check_attr=True):
        scene_name = robot.root.name
        print(colorama.Fore.BLUE + colorama.Style.BRIGHT + scene_name)
        results = {}

        # robot.config["init_frames"] = None
        for i in (init_frames or robot.config["init_frames"] or list(range(len(robot)))):
            for s in seeds:
                trial = f"{scene_name}/init{i}/seed{s}"
                robot.set_cur_frame(i, seed=s)
                metrics = pd.DataFrame(columns=["AP", "pre_greedy"])

                # iterate
                scene = Scene(system.config, **robot.kwargs_scene)
                for j in range(max_iter):
                    folder = f"{trial}/iter{j}"

                    # forward
                    frame = robot.get_frame()
                    if frame.Tcw is scene.Tcw: break
                    frame.metadata["seed"] = s
                    scene.integrate(frame)
                    act = system.planner.solve(scene) if j + 1 != max_iter else None

                    # save
                    scene.save_info(
                        folder=logdir.pdir(folder),
                        tog=True, traj=True, field=not system.cfg_nbv_proxy
                    )

                    # evaluate
                    tog = scene.tog_poses[0]
                    y_true = robot.tog_label[robot.tog_label.nearest_search(tog)].score
                    y_score = tog.score
                    logdir.save_data((tog, y_true), f"{folder}/tog_and_true.pkl")
                    metrics.loc[j] = [
                        skm.average_precision_score(y_true, y_score).item() if y_true.any() else 0.,
                        y_true[0]
                    ]

                    # execute
                    if not act: break
                    robot.move(act.Teb)

                # save metrics
                print(colorama.Fore.CYAN + f"{trial}: " + colorama.Style.RESET_ALL + str(metrics.to_dict(orient='list')))
                for j in range(len(metrics), max_iter):
                    metrics.loc[j] = metrics.iloc[-1]
                results[trial] = metrics

        # aggregate
        results["avg"] = sum(results.values()) / len(results)
        results = {k: v.to_dict(orient='list') for k, v in results.items()}
        logdir.save_data(results, f"{scene_name}/results.yaml")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="gcngrasp-vp", help="project name")
    parser.add_argument("--config", type=str, default="config/system/params.yaml", help="config file")
    parser.add_argument("--data-path", type=str, default="./dataset", help="data directory")
    parser.add_argument("--max-iter", type=int, default=4, help="max iteration")
    parser.add_argument("--n-seed", type=int, default=4, help="number of seeds")
    parser.add_argument("--init-frames", type=int, nargs="+", help="init frames")
    parser.add_argument("--vis-path", type=str, help="visualize directory")

    args = parser.parse_args()
    CONFIG = Path(args.config)

    if args.vis_path:
        for f in Path(args.vis_path).rglob("visualize.glb"):
            print(f)
            scene = trimesh.load(f)
            scene.show()

    else:
        cfg_planner = dict(
            border_margin=.05,
            max_stride=.50,
            min_stride=.10,
            n_candidates=50,
            dist_obs=0.,
            elev_range=None,
        )

        cfg = process_config(CONFIG, with_api=True)
        cfg["planner"].update(cfg_planner)

        system = System(cfg, cfg_nbv_proxy=None if args.name == "gcngrasp-vp" else cfg_planner)
        logdir = LogDir(args.name, rootdir="runs/nbv_logs")
        run(
            system, logdir,
            data_path=Path(args.data_path),
            max_iter=args.max_iter,
            seeds=list(range(args.n_seed)),
            init_frames=args.init_frames,
        )
